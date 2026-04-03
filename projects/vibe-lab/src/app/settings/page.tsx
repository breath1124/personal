import { AppShell } from "@/components/app-shell";
import { SettingsPanel } from "@/components/settings-panel";

export default function SettingsPage() {
  return (
    <AppShell
      activePath="/settings"
      title="模型设置"
      description="默认按 OpenAI 兼容接口工作，也支持替换成你自己的 Base URL 和模型。"
    >
      <SettingsPanel />
    </AppShell>
  );
}
