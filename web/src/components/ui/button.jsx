import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "kitchen-focus inline-flex h-10 shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-md border px-4 text-sm font-black uppercase tracking-normal transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "border-primary/70 bg-primary/18 text-primary shadow-[0_0_26px_rgba(57,255,154,0.22)] hover:bg-primary/28 hover:shadow-[0_0_34px_rgba(57,255,154,0.34)]",
        secondary:
          "border-border bg-card/80 text-foreground hover:border-primary/55 hover:bg-primary/10",
        destructive:
          "border-destructive/60 bg-destructive/12 text-destructive hover:bg-destructive/20",
        ghost:
          "border-transparent bg-transparent text-muted-foreground hover:border-border hover:bg-primary/10 hover:text-primary",
        amber:
          "border-amber/65 bg-amber/14 text-amber shadow-[0_0_24px_rgba(255,200,87,0.18)] hover:bg-amber/22",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-8 px-3 text-xs",
        lg: "h-12 px-6 text-base",
        icon: "h-10 w-10 p-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

function Button({ className, variant, size, asChild = false, ...props }) {
  const Comp = asChild ? Slot : "button";
  return <Comp className={cn(buttonVariants({ variant, size, className }))} {...props} />;
}

export { Button, buttonVariants };
