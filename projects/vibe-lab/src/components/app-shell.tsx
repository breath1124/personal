import Link from "next/link";
import { ReactNode } from "react";

const links = [
  { href: "/", label: "Lab 首页" },
  { href: "/fund", label: "基金助手" },
  { href: "/mbti", label: "MBTI" },
  { href: "/reaction", label: "反应力" },
  { href: "/settings", label: "设置" }
];

export function AppShell({
  activePath,
  title,
  description,
  children
}: {
  activePath: string;
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <Link className="brand-mark" href="/">
          CY Lab
        </Link>
        <nav className="topnav" aria-label="实验室导航">
          {links.map((link) => (
            <Link
              key={link.href}
              className={`topnav-link${activePath === link.href ? " is-active" : ""}`}
              href={link.href}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </header>

      <main className="page-frame">
        <section className="page-hero">
          <p className="eyebrow">Vibe Coding Lab</p>
          <h1>{title}</h1>
          <p>{description}</p>
        </section>
        {children}
      </main>
    </div>
  );
}
