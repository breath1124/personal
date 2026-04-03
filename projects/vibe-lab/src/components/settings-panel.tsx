"use client";

import { useState } from "react";
import { DEFAULT_AI_SETTINGS, AI_SETTINGS_KEY } from "@/lib/config";
import { requestModelAnalysis } from "@/lib/openai";
import { usePersistentState } from "@/lib/storage";

export function SettingsPanel() {
  const [settings, setSettings, hydrated] = usePersistentState(
    AI_SETTINGS_KEY,
    DEFAULT_AI_SETTINGS
  );
  const [status, setStatus] = useState<string>("本地保存，仅当前浏览器可见。");
  const [testing, setTesting] = useState(false);

  async function testConnection() {
    setTesting(true);
    setStatus("正在连接模型...");

    try {
      const reply = await requestModelAnalysis(settings, [
        {
          role: "system",
          content: "You are a concise system check assistant."
        },
        {
          role: "user",
          content: "请只回复：连接成功。"
        }
      ]);
      setStatus(`连接成功：${reply}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "连接失败");
    } finally {
      setTesting(false);
    }
  }

  function resetSettings() {
    setSettings(DEFAULT_AI_SETTINGS);
    setStatus("已恢复默认 OpenAI 配置。");
  }

  return (
    <div className="section-stack">
      <section className="panel-card">
        <div className="section-heading section-heading--tight">
          <div>
            <p className="eyebrow">Provider</p>
            <h2>模型连接设置</h2>
          </div>
          <span className="muted">{hydrated ? "已同步本地配置" : "读取配置中"}</span>
        </div>

        <div className="form-grid">
          <label className="field">
            <span>显示名称</span>
            <input
              className="control"
              value={settings.providerName}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  providerName: event.target.value
                }))
              }
            />
          </label>

          <label className="field">
            <span>Base URL</span>
            <input
              className="control"
              value={settings.baseUrl}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  baseUrl: event.target.value
                }))
              }
            />
          </label>

          <label className="field">
            <span>Model</span>
            <input
              className="control"
              value={settings.model}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  model: event.target.value
                }))
              }
            />
          </label>

          <label className="field">
            <span>Temperature</span>
            <input
              className="control"
              max="1.5"
              min="0"
              step="0.1"
              type="number"
              value={settings.temperature}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  temperature: Number(event.target.value)
                }))
              }
            />
          </label>
        </div>

        <label className="field field--full">
          <span>API Key</span>
          <input
            className="control"
            placeholder="sk-..."
            type="password"
            value={settings.apiKey}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                apiKey: event.target.value
              }))
            }
          />
        </label>

        <div className="button-row">
          <button className="button" disabled={!settings.apiKey || testing} onClick={testConnection}>
            {testing ? "连接中..." : "测试连接"}
          </button>
          <button className="button button--secondary" onClick={resetSettings}>
            恢复默认值
          </button>
        </div>

        <p className="callout">{status}</p>
      </section>

      <section className="panel-card">
        <div className="section-heading section-heading--tight">
          <div>
            <p className="eyebrow">Privacy</p>
            <h2>使用方式</h2>
          </div>
        </div>
        <ul className="bullet-list">
          <li>当前应用不会把你的 API Key 写入仓库，也不会上传到我的环境。</li>
          <li>所有配置默认保存在浏览器本地，清空浏览器数据后会丢失。</li>
          <li>如果你想切换到其他兼容接口，只需要改 Base URL 和 Model。</li>
        </ul>
      </section>
    </div>
  );
}
