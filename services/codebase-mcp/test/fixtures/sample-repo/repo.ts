// repo.ts — data layer leaf of the route -> handler -> service -> repo chain.
// No outgoing call edges from the chain's perspective (terminal node).

export interface UserRow {
  id: string;
  name: string;
}

export function findUser(id: string): UserRow {
  return { id, name: "sample" };
}
