import { AppShell } from "@/components/app-shell";
import { ReactionLab } from "@/components/reaction-lab";

export default function ReactionPage() {
  return (
    <AppShell
      activePath="/reaction"
      title="反应力实验室"
      description="用三段测试测启动速度、选择反应和抑制控制，再给你一份带逐轮数据的详细分析。"
    >
      <ReactionLab />
    </AppShell>
  );
}
