export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch("/api" + path, options);
  } catch {
    throw new Error("无法连接本机服务，请检查服务是否已启动。");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.detail ?? body?.message;
    throw new ApiError(response.status,
      typeof message === "string"
        ? message
        : Array.isArray(message)
          ? message.map((x) => x.msg).join("，")
          : "请求失败（HTTP " + response.status + "）",
    );
  }
  return response.json();
}
export function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
export const paperUrl = (id: string) => "/api/papers/" + encodeURIComponent(id);
export const errorMessage = (err: unknown): string =>
  err instanceof Error ? err.message : String(err);
