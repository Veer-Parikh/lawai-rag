"use client";

import { useRouter } from "next/navigation";
import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { fetchMe, readStoredSession, Role, Session, signInAdmin, signInDemo, storeSession } from "./api";

type AuthState = {
  session: Session | null;
  ready: boolean;
  continueAsDemo: () => Promise<void>;
  signInAsAdmin: (password: string) => Promise<void>;
  signOut: () => void;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const stored = readStoredSession();
    setSession(stored);
    setReady(true);
    if (!stored) return;
    // The API may have restarted with a new signing secret; drop stale sessions.
    fetchMe().catch((error) => {
      if (error?.status === 401) {
        storeSession(null);
        setSession(null);
      }
    });
  }, []);

  const adopt = useCallback((next: Session | null) => {
    storeSession(next);
    setSession(next);
  }, []);

  const value = useMemo<AuthState>(
    () => ({
      session,
      ready,
      continueAsDemo: async () => adopt(await signInDemo()),
      signInAsAdmin: async (password) => adopt(await signInAdmin(password)),
      signOut: () => adopt(null),
    }),
    [session, ready, adopt],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}

/** Redirects away when there is no session, or the session lacks the role. */
export function useRequireRole(role?: Role) {
  const { session, ready } = useAuth();
  const router = useRouter();
  const allowed = !!session && (!role || session.role === role);

  useEffect(() => {
    if (!ready) return;
    if (!session) router.replace("/login");
    else if (role && session.role !== role) router.replace("/research");
  }, [ready, session, role, router]);

  return { session, allowed: ready && allowed };
}
