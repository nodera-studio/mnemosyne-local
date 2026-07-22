// service.ts — third hop. Imports repo.ts and calls findUser.
import { findUser, type UserRow } from "./repo.js";

export function getUser(id: string): UserRow {
  return findUser(id);
}
