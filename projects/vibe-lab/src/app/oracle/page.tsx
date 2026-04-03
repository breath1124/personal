import { AppShell } from "@/components/app-shell";
import { OracleStudio } from "@/components/oracle-studio";

export default function OraclePage() {
  return (
    <AppShell
      activePath="/oracle"
      title="命理实验室"
      description="用八字实验盘和紫微斗数镜像看底色，再把神秘叙事翻译成现实问题、观察信号和可执行动作。"
    >
      <OracleStudio />
    </AppShell>
  );
}
