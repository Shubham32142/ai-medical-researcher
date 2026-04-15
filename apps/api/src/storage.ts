import dns from "node:dns/promises";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import mongoose from "mongoose";
import type { Message, Session } from "@curalink/types";

const dataFile = join(process.cwd(), "data", "sessions.json");
const localSessions = new Map<string, Session>();
let mongoEnabled = false;

const MessageSchema = new mongoose.Schema<Message>(
  {
    id: String,
    role: String,
    content: String,
    citations: { type: Array, default: [] },
    trials: { type: Array, default: [] },
    followUps: { type: Array, default: [] },
    createdAt: String,
  },
  { _id: false },
);

const SessionSchema = new mongoose.Schema<Session>(
  {
    sessionId: { type: String, unique: true, index: true },
    disease: String,
    location: String,
    age: Number,
    comorbidities: { type: [String], default: [] },
    messages: { type: [MessageSchema], default: [] },
    createdAt: String,
    updatedAt: String,
  },
  { versionKey: false },
);

const SessionModel =
  (mongoose.models.CuraLinkSession as mongoose.Model<Session>) ||
  mongoose.model<Session>("CuraLinkSession", SessionSchema);

async function ensureLocalFile() {
  await mkdir(dirname(dataFile), { recursive: true });
  try {
    await readFile(dataFile, "utf8");
  } catch {
    await writeFile(dataFile, "[]", "utf8");
  }
}

async function loadLocalSessions() {
  await ensureLocalFile();
  const raw = await readFile(dataFile, "utf8");
  const parsed = JSON.parse(raw || "[]") as Session[];
  localSessions.clear();
  for (const session of parsed) {
    localSessions.set(session.sessionId, session);
  }
}

async function persistLocalSessions() {
  await ensureLocalFile();
  const payload = JSON.stringify(Array.from(localSessions.values()), null, 2);
  await writeFile(dataFile, payload, "utf8");
}

function normalizeSession(session: Session): Session {
  return {
    ...session,
    messages: session.messages ?? [],
    comorbidities: session.comorbidities ?? [],
  };
}

export async function initializePersistence() {
  const mongoUri =
    process.env.MONGODB_DIRECT_URI?.trim() || process.env.MONGODB_URI?.trim();

  if (mongoUri) {
    try {
      if (mongoUri.startsWith("mongodb+srv://")) {
        dns.setServers(["1.1.1.1", "8.8.8.8"]);
      }
      await mongoose.connect(mongoUri, { serverSelectionTimeoutMS: 4000 });
      mongoEnabled = true;
      return { mode: "mongo" as const };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (
        mongoUri.startsWith("mongodb+srv://") &&
        /querySrv|ECONNREFUSED/i.test(message)
      ) {
        console.warn(
          "MongoDB SRV DNS lookup failed in the runtime. If Atlas works in Compass, paste the direct mongodb:// connection string into MONGODB_DIRECT_URI.",
        );
      }
      console.warn(
        "MongoDB unavailable, using local persistent storage.",
        error,
      );
    }
  }

  await loadLocalSessions();
  mongoEnabled = false;
  return { mode: "local" as const };
}

export async function listSessions(): Promise<Session[]> {
  if (mongoEnabled) {
    const docs = await SessionModel.find().lean<Session[]>();
    return docs
      .map(normalizeSession)
      .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1));
  }

  return Array.from(localSessions.values())
    .map(normalizeSession)
    .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1));
}

export async function getSession(sessionId: string): Promise<Session | null> {
  if (mongoEnabled) {
    const doc = await SessionModel.findOne({
      sessionId,
    }).lean<Session | null>();
    return doc ? normalizeSession(doc) : null;
  }

  return localSessions.get(sessionId) ?? null;
}

export async function createSession(session: Session): Promise<Session> {
  if (mongoEnabled) {
    await SessionModel.create(session);
    return session;
  }

  localSessions.set(session.sessionId, session);
  await persistLocalSessions();
  return session;
}

export async function saveSession(session: Session): Promise<Session> {
  if (mongoEnabled) {
    await SessionModel.updateOne({ sessionId: session.sessionId }, session, {
      upsert: true,
    });
    return session;
  }

  localSessions.set(session.sessionId, session);
  await persistLocalSessions();
  return session;
}

export async function appendMessage(
  sessionId: string,
  message: Message,
): Promise<Session | null> {
  const session = await getSession(sessionId);
  if (!session) return null;

  session.messages.push(message);
  session.updatedAt = message.createdAt;
  await saveSession(session);
  return session;
}
