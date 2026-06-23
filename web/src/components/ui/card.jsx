import * as React from "react";

import { cn } from "@/lib/utils";

function Card({ className, ...props }) {
  return (
    <div
      className={cn(
        "rounded-md border border-border bg-card/88 text-card-foreground shadow-[0_22px_60px_rgba(0,0,0,0.36)] backdrop-blur-xl",
        "before:pointer-events-none before:absolute before:inset-0 before:rounded-md before:border before:border-white/4",
        "relative overflow-hidden",
        className,
      )}
      {...props}
    />
  );
}

function CardHeader({ className, ...props }) {
  return <div className={cn("flex flex-col gap-1.5 p-4", className)} {...props} />;
}

function CardTitle({ className, ...props }) {
  return (
    <h3
      className={cn("text-sm font-black uppercase leading-none tracking-normal", className)}
      {...props}
    />
  );
}

function CardDescription({ className, ...props }) {
  return <p className={cn("text-xs text-muted-foreground", className)} {...props} />;
}

function CardContent({ className, ...props }) {
  return <div className={cn("p-4 pt-0", className)} {...props} />;
}

function CardFooter({ className, ...props }) {
  return <div className={cn("flex items-center p-4 pt-0", className)} {...props} />;
}

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent };
