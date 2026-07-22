// handler.ts — second hop. Imports service.ts and calls getUser.
import { getUser } from "./service.js";
import type { UserRow } from "./repo.js";

export function handleGetUser(id: string): UserRow {
  return getUser(id);
}
