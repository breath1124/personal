import type { Metadata } from "next";
import { Manrope } from "next/font/google";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap"
});

export const metadata: Metadata = {
  title: "CY Lab",
  description: "更成熟的 Vibe Coding 应用区，包含基金助手、MBTI 助手与反应力实验室。"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className={manrope.variable}>{children}</body>
    </html>
  );
}
