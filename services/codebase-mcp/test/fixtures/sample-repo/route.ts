// route.ts — entry point of the chain. Imports handler.ts and calls handleGetUser.
// This is the seed symbol used for the depth-1/2/4 traversal expectation table.
import { handleGetUser } from "./handler.js";

export function route(id: string): string {
  const user = handleGetUser(id);
  return user.name;
}
