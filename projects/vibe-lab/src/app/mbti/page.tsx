import { AppShell } from "@/components/app-shell";
import { MbtiAssistant } from "@/components/mbti-assistant";

export default function MbtiPage() {
  return (
    <AppShell
      activePath="/mbti"
      title="MBTI 助手"
      description="用更完整的题组、维度结果和场景化建议，得到一份真正能拿来做沟通和工作复盘的报告。"
    >
      <MbtiAssistant />
    </AppShell>
  );
}
