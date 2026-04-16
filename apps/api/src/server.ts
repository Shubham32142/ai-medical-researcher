import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import type { ChatResponse, Message, Session } from "@curalink/types";
import {
  createSession,
  getSession,
  initializePersistence,
  listSessions,
  saveSession,
} from "./storage";

dotenv.config({ path: ".env.local", override: true });
dotenv.config();

const app = express();
const port = Number(process.env.PORT || 4000);
const aiServiceUrl = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";
const allowedOrigins = (
  process.env.CORS_ORIGIN || "http://localhost:5173,http://127.0.0.1:5173"
)
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const sessionSchema = z.object({
  disease: z.string().min(2),
  location: z.string().min(2),
  age: z.number().int().positive().optional(),
  comorbidities: z.array(z.string()).optional().default([]),
});

const chatSchema = z.object({
  sessionId: z.string().min(3),
  message: z.string().min(2),
});

app.use(
  cors({
    origin(origin, callback) {
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin)) return callback(null, true);
      if (/\.vercel\.app$/i.test(origin)) return callback(null, true);
      return callback(new Error(`CORS blocked for origin: ${origin}`));
    },
    credentials: true,
  }),
);
app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "api" });
});

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", service: "api", aiServiceUrl });
});

app.get("/api/sessions", async (_req, res) => {
  const list = (await listSessions()).map((session) => ({
    sessionId: session.sessionId,
    disease: session.disease,
    location: session.location,
    lastMessageAt: session.updatedAt,
  }));

  res.json({ sessions: list });
});

app.post("/api/sessions", async (req, res) => {
  const parsed = sessionSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }

  const now = new Date().toISOString();
  const session: Session = {
    sessionId: `ses_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
    disease: parsed.data.disease,
    location: parsed.data.location,
    age: parsed.data.age,
    comorbidities: parsed.data.comorbidities,
    messages: [],
    createdAt: now,
    updatedAt: now,
  };

  await createSession(session);
  res.status(201).json(session);
});

app.get("/api/sessions/:id", async (req, res) => {
  const session = await getSession(req.params.id);
  if (!session) {
    return res.status(404).json({ error: "Session not found" });
  }

  res.json(session);
});

function writeEvent(res: express.Response, event: string, data: unknown) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function buildFallback(session: Session, message: string): ChatResponse {
  return {
    answer:
      `I could not reach the research service, so this is a limited fallback response for ${session.disease}. ` +
      `Please retry once the AI service is running. Your latest question was: ${message}`,
    citations: [],
    trials: [],
    followUps: [
      `What are the latest trials for ${session.disease}?`,
      `What should I ask my doctor about ${session.disease}?`,
    ],
  };
}

app.post("/api/chat", async (req, res) => {
  const parsed = chatSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }

  const session = await getSession(parsed.data.sessionId);
  if (!session) {
    return res.status(404).json({ error: "Session not found" });
  }

  const userMessage: Message = {
    id: `msg_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
    role: "user",
    content: parsed.data.message,
    createdAt: new Date().toISOString(),
  };

  session.messages.push(userMessage);
  session.updatedAt = userMessage.createdAt;
  await saveSession(session);

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin":
      req.headers.origin || allowedOrigins[0] || "*",
  });

  writeEvent(res, "status", { stage: "fetching_sources" });

  let result: ChatResponse;
  try {
    const aiResponse = await fetch(`${aiServiceUrl}/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: session.sessionId,
        disease: session.disease,
        location: session.location,
        history: session.messages.slice(-8),
        message: parsed.data.message,
      }),
    });

    if (!aiResponse.ok) {
      throw new Error(`AI service returned ${aiResponse.status}`);
    }

    result = (await aiResponse.json()) as ChatResponse;
  } catch (error) {
    console.error(error);
    result = buildFallback(session, parsed.data.message);
  }

  writeEvent(res, "status", { stage: "synthesizing" });

  let assembled = "";
  const tokens = result.answer.split(/(\s+)/).filter(Boolean);
  for (const token of tokens) {
    assembled += token;
    writeEvent(res, "token", { delta: token });
    await new Promise((resolve) => setTimeout(resolve, 8));
  }

  writeEvent(res, "citations", { citations: result.citations });
  writeEvent(res, "trials", { trials: result.trials });
  writeEvent(res, "followups", { followUps: result.followUps });

  const assistantMessage: Message = {
    id: `msg_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
    role: "assistant",
    content: assembled,
    citations: result.citations,
    trials: result.trials,
    followUps: result.followUps,
    createdAt: new Date().toISOString(),
  };

  session.messages.push(assistantMessage);
  session.updatedAt = assistantMessage.createdAt;
  await saveSession(session);

  writeEvent(res, "done", { messageId: assistantMessage.id });
  res.end();
});

initializePersistence()
  .then(({ mode }) => {
    app.listen(port, () => {
      console.log(
        `CuraLink API listening on http://localhost:${port} using ${mode} persistence`,
      );
    });
  })
  .catch((error) => {
    console.error("Failed to initialize persistence", error);
    process.exit(1);
  });
