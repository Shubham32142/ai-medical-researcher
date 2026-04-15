export interface Citation {
  id: string;
  source: "pubmed" | "openalex";
  title: string;
  authors: string[];
  journal?: string;
  year: number;
  abstract: string;
  doi?: string;
  url: string;
  relevanceScore: number;
}

export interface TrialLocation {
  facility: string;
  city: string;
  country: string;
}

export interface Trial {
  nctId: string;
  title: string;
  status: string;
  phase?: string;
  conditions: string[];
  interventions: string[];
  locations: TrialLocation[];
  url: string;
  nearUser?: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Citation[];
  trials?: Trial[];
  followUps?: string[];
  createdAt: string;
}

export interface Session {
  sessionId: string;
  disease: string;
  location: string;
  age?: number;
  comorbidities?: string[];
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  trials: Trial[];
  followUps: string[];
}
