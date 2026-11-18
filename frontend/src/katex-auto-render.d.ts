declare module "katex/contrib/auto-render" {
  export default function renderMathInElement(
    el: Element,
    options: {
      throwOnError: boolean;
      trust: boolean;
      strict: string;
      delimiters: { left: string; right: string; display: boolean }[];
    },
  ): void;
}
