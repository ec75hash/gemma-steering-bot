import * as React from "react";

import { cn } from "@/lib/utils";

const Textarea = React.forwardRef(({ className, ...props }, ref) => {
  return (
    <textarea
      className={cn(
        "kitchen-focus flex min-h-[80px] w-full rounded-md border border-input bg-background/75 px-3 py-2 text-sm font-bold leading-relaxed text-foreground shadow-inner shadow-black/35 placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      ref={ref}
      {...props}
    />
  );
});
Textarea.displayName = "Textarea";

export { Textarea };
