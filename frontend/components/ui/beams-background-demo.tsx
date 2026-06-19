import { BeamsBackground } from "@/components/ui/beams-background";

export function BeamsBackgroundDemo() {
  return (
    <BeamsBackground intensity="medium">
      <div className="flex min-h-screen w-full items-center justify-center px-4">
        <div className="text-center">
          <h1 className="text-5xl font-semibold tracking-tighter text-white md:text-7xl">
            VioLane
          </h1>
          <p className="mt-4 text-lg text-white/70 md:text-2xl">
            Traffic violation analytics
          </p>
        </div>
      </div>
    </BeamsBackground>
  );
}
