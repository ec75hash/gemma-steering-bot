import * as React from "react";
import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-black uppercase tracking-normal transition-colors",
  {
    variants: {
      variant: {
        default: "border-primary/65 bg-primary/14 text-primary",
        secondary: "border-border bg-secondary text-secondary-foreground",
        amber: "border-amber/65 bg-amber/14 text-amber",
        destructive: "border-destructive/65 bg-destructive/12 text-destructive",
        outline: "border-border text-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

function Badge({ className, variant, ...props }) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
