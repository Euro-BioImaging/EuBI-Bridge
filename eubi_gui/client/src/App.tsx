import { Switch, Route } from "wouter";
import { useEffect } from "react";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import NotFound from "@/pages/not-found";
import ConvertPage from "@/pages/convert-page";
import InspectPage from "@/pages/inspect-page";
import { useApplyConfig } from "@/hooks/use-config";

/** Fetches ~/.eubi_bridge/.eubi_config.json once on startup and hydrates the store. */
function ConfigLoader() {
  const applyConfig = useApplyConfig();
  useEffect(() => {
    fetch("/api/config")
      .then((res) => {
        if (!res.ok || !res.headers.get("content-type")?.includes("application/json")) return;
        return res.json();
      })
      .then((data) => {
        if (data) applyConfig(data);
      })
      .catch(() => {
        // Silently ignore — server may not be ready yet or file may not exist
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

function Router() {
  return (
    <Switch>
      <Route path="/" component={ConvertPage} />
      <Route path="/inspect" component={InspectPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

const sidebarStyle = {
  "--sidebar-width": "18rem",
  "--sidebar-width-icon": "3rem",
};

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <ConfigLoader />
          <SidebarProvider style={sidebarStyle as React.CSSProperties}>
            <div className="flex h-screen w-full">
              <AppSidebar />
              <div className="flex flex-col flex-1 min-w-0">
                <header className="flex items-center justify-between gap-1 px-4 py-2 border-b bg-background/80 backdrop-blur-sm sticky top-0 z-50">
                  <div className="flex items-center gap-2">
                    <SidebarTrigger data-testid="button-sidebar-toggle" />
                  </div>
                  <ThemeToggle />
                </header>
                <main className="flex-1 overflow-y-auto">
                  <div className="max-w-6xl mx-auto p-6">
                    <Router />
                  </div>
                </main>
              </div>
            </div>
          </SidebarProvider>
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
