"use client";

import { CircleNotch } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { useAuth } from "../auth";
import { errorText } from "../_components/ui";

export default function LoginPage() {
  const { session, ready, continueAsDemo, signInAsAdmin } = useAuth();
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState<"demo" | "admin" | null>(null);
  const [demoError, setDemoError] = useState("");
  const [adminError, setAdminError] = useState("");

  useEffect(() => {
    if (ready && session) router.replace(session.role === "admin" ? "/admin" : "/research");
  }, [ready, session, router]);

  async function demo() {
    setPending("demo");
    setDemoError("");
    try {
      await continueAsDemo();
    } catch (error) {
      setDemoError(errorText(error, "Demo sign-in failed."));
    } finally {
      setPending(null);
    }
  }

  async function admin(event: FormEvent) {
    event.preventDefault();
    if (!password) {
      setAdminError("Enter the administrator password.");
      return;
    }
    setPending("admin");
    setAdminError("");
    try {
      await signInAsAdmin(password);
    } catch (error) {
      setAdminError(errorText(error, "Sign-in failed."));
    } finally {
      setPending(null);
    }
  }

  return (
    <main id="main" className="login">
      <section className="login-specimen" aria-label="How LawAI shows a provision">
        <p className="specimen-act">Section 103 of the BNS replaced section 302 of the IPC on 1 July 2024.</p>
        <div className="specimen">
          <span className="specimen-no">103.</span>
          <p className="specimen-text">
            (1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be
            liable to fine.
          </p>
          <div className="specimen-margin">
            <span className="mark">Formerly IPC, section 302</span>
            <span className="tag tag-ok">In force from 1 July 2024</span>
            <span>Notification S.O. 850(E), Gazette of India</span>
          </div>
        </div>
        <p className="lede">Every answer in LawAI comes with the provision, what it replaced, when it took effect, and the instrument that proves it.</p>
      </section>

      <section className="login-panel">
        <span className="wordmark">LawAI</span>
        <h1>Sign in</h1>
        <p className="lede">Research the BNS, BNSS and BSA, and the laws they replaced.</p>

        <div className="login-block">
          <button type="button" className="btn btn-primary" onClick={demo} disabled={pending !== null}>
            {pending === "demo" && <CircleNotch className="spin" size={18} aria-hidden="true" />}
            Continue as demo researcher
          </button>
          <p className="hint">No account needed. Your saved matters stay with this browser.</p>
          {demoError && <p className="field-error" role="alert">{demoError}</p>}
        </div>

        <form className="login-block" onSubmit={admin} noValidate>
          <h2>Administrators</h2>
          <label htmlFor="admin-password">Admin password</label>
          <input
            id="admin-password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            aria-invalid={!!adminError}
            aria-describedby={adminError ? "admin-error" : "admin-hint"}
          />
          <p className="hint" id="admin-hint">Set as ADMIN_PASSWORD in the API environment.</p>
          {adminError && <p className="field-error" id="admin-error" role="alert">{adminError}</p>}
          <button type="submit" className="btn" disabled={pending !== null}>
            {pending === "admin" && <CircleNotch className="spin" size={18} aria-hidden="true" />}
            Sign in as administrator
          </button>
        </form>

        <p className="login-foot">Research support, not legal advice.</p>
      </section>
    </main>
  );
}
