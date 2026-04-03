export function formatDate(value: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleDateString("zh-CN");
}

export function formatDateTime(value: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN");
}

export function formatPct(value: number | null, digits = 2) {
  if (value === null || Number.isNaN(value)) return "-";
  return `${value.toFixed(digits)}%`;
}

export function formatNumber(value: number | null, digits = 2) {
  if (value === null || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

export function formatCurrency(value: number | null) {
  if (value === null || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 2
  }).format(value);
}
