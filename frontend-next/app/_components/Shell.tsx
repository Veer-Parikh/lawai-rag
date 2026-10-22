"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ReactNode } from "react";
import { Role } from "../api";
import { useAuth, useRequireRole } from "../auth";

const PRIMARY = [
  { href: "/research", label: "Research" },
  { href: "/crosswalk", label: "Crosswalk" },
  { href: "/timeline", label: "Timeline" },
  { href: "/changes", label: "Changes" },
  { href: "/matters", label: "Matters" },
];

const ADMIN = [
  { href: "/admin", label: "Overview" },
  { href: "/admin/corpus", label: "Corpus" },
  { href: "/admin/sources", label: "Sources" },
  { href: "/admin/review", label: "Review queue" },
];

function isActive(pathname: string, href: string) {
  if (href === "/" || href === "/admin") return pathname === href;
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function Shell({ children, role }: { children: ReactNode; role?: Role }) {
  const { session, allowed } = useRequireRole(role);
  const { signOut } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  if (!allowed || !session) {
    return <div className="boot" role="status">Checking your session…</div>;
  }

  const inAdmin = pathname.startsWith("/admin");

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <Link href="/" className="wordmark">LawAI</Link>
          <nav className="nav" aria-label="Main">
            {PRIMARY.map((item) => (
              <Link key={item.href} href={item.href} className="nav-link" aria-current={isActive(pathname, item.href) ? "page" : undefined}>
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="topbar-end">
            {session.role === "admin" && (
              <Link href="/admin" className="nav-link" aria-current={inAdmin ? "page" : undefined}>Admin</Link>
            )}
            <span className="user-name">{session.name}</span>
            <button
              type="button"
              className="btn-quiet"
              onClick={() => {
                signOut();
                router.replace("/login");
              }}
            >
              Sign out
            </button>
          </div>
        </div>
      </header>
      {inAdmin && session.role === "admin" && (
        <div className="subnav">
          <nav className="subnav-inner" aria-label="Administration">
            <span className="subnav-title">Administration</span>
            {ADMIN.map((item) => (
              <Link key={item.href} href={item.href} className="nav-link" aria-current={isActive(pathname, item.href) ? "page" : undefined}>
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      )}
      <main id="main" className="page">{children}</main>
    </>
  );
}
