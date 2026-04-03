"use client";

import { useEffect, useState } from "react";

export function readStorage<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;

  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function writeStorage<T>(key: string, value: T) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, JSON.stringify(value));
}

export function usePersistentState<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(fallback);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setValue(readStorage(key, fallback));
    setHydrated(true);
  }, [fallback, key]);

  useEffect(() => {
    if (!hydrated) return;
    writeStorage(key, value);
  }, [hydrated, key, value]);

  return [value, setValue, hydrated] as const;
}
