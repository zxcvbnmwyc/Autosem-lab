(() => {
  "use strict";

  const STORAGE_KEY = "autosem.ops.token";
  const endpoint = "/api/ops/metrics";

  const state = {
    token: "",
    windowHours: 24,
    loading: false,
  };

  const elements = {
    tokenForm: document.querySelector("#token-form"),
    tokenInput: document.querySelector("#ops-token"),
    tokenSubmit: document.querySelector("#token-submit"),
    tokenVisibility: document.querySelector("#token-visibility"),
    authCard: document.querySelector("#auth-card"),
    authMessage: document.querySelector("#auth-message"),
    refreshButton: document.querySelector("#refresh-button"),
    periodButtons: [...document.querySelectorAll(".period-button")],
    updatedAt: document.querySelector("#updated-at"),
    windowCopy: document.querySelector("#window-copy"),
    status: document.querySelector("#connection-status"),
    notice: document.querySelector("#notice"),
    metricGrid: document.querySelector(".metric-grid"),
    eventsList: document.querySelector("#events-list"),
    eventsEmpty: document.querySelector("#events-empty"),
  };

  const metricElements = {
    sessions: document.querySelector("#metric-sessions"),
    pageViews: document.querySelector("#metric-page-views"),
    uploads: document.querySelector("#metric-uploads"),
    segmentJobs: document.querySelector("#metric-segment-jobs"),
    segmentNote: document.querySelector("#metric-segment-note"),
    qwen: document.querySelector("#metric-qwen"),
    agent: document.querySelector("#metric-agent"),
  };

  const timingElements = {
    qwen: {
      median: document.querySelector("#qwen-median"),
      p90: document.querySelector("#qwen-p90"),
      count: document.querySelector("#qwen-count"),
    },
    sam2: {
      median: document.querySelector("#sam2-median"),
      p90: document.querySelector("#sam2-p90"),
      count: document.querySelector("#sam2-count"),
    },
    sam2Core: {
      card: document.querySelector("#sam2-core-card"),
      median: document.querySelector("#sam2-core-median"),
      p90: document.querySelector("#sam2-core-p90"),
      count: document.querySelector("#sam2-core-count"),
    },
    queue: {
      median: document.querySelector("#queue-median"),
      p90: document.querySelector("#queue-p90"),
      count: document.querySelector("#queue-count"),
    },
  };

  function number(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function count(value) {
    const numeric = number(value);
    return numeric === null ? "—" : new Intl.NumberFormat("zh-CN").format(numeric);
  }

  function percent(value) {
    const numeric = number(value);
    if (numeric === null) return "暂无成功率数据";
    const normalized = numeric <= 1 ? numeric * 100 : numeric;
    return `成功率 ${new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(normalized)}%`;
  }

  function duration(value) {
    const numeric = number(value);
    if (numeric === null || numeric < 0) return "—";
    if (numeric < 1000) return `${Math.round(numeric)} ms`;
    if (numeric < 10000) return `${(numeric / 1000).toFixed(2)} 秒`;
    return `${(numeric / 1000).toFixed(1)} 秒`;
  }

  function time(value) {
    if (!value) return "刚刚更新";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "刚刚更新";
    return parsed.toLocaleString("zh-CN", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function windowLabel(hours) {
    if (hours === 24) return "近 24 小时";
    if (hours === 168) return "近 7 天";
    if (hours === 720) return "近 30 天";
    return `近 ${hours} 小时`;
  }

  function setConnection(kind, label) {
    const status = elements.status;
    status.className = `connection-status connection-status--${kind}`;
    const text = status.querySelector("span");
    if (text) text.textContent = label;
  }

  function setNotice(message, kind = "info") {
    const notice = elements.notice;
    if (!message) {
      notice.hidden = true;
      notice.textContent = "";
      notice.className = "notice";
      return;
    }
    notice.hidden = false;
    notice.className = `notice notice--${kind}`;
    notice.textContent = message;
  }

  function setLoading(loading) {
    state.loading = loading;
    elements.refreshButton.disabled = loading || !state.token;
    elements.tokenSubmit.disabled = loading;
    elements.metricGrid.setAttribute("aria-busy", String(loading));
    elements.refreshButton.classList.toggle("is-loading", loading);
    elements.refreshButton.querySelector(".refresh-icon").textContent = loading ? "↻" : "↻";
  }

  function refreshPeriodControls() {
    for (const button of elements.periodButtons) {
      const selected = Number(button.dataset.windowHours) === state.windowHours;
      button.setAttribute("aria-pressed", String(selected));
      button.classList.toggle("is-selected", selected);
    }
    elements.windowCopy.textContent = `${windowLabel(state.windowHours)}的汇总数据`;
  }

  function renderTiming(target, data) {
    const timing = data && typeof data === "object" ? data : {};
    target.median.textContent = duration(timing.median_ms);
    target.p90.textContent = duration(timing.p90_ms);
    target.count.textContent = count(timing.count);
  }

  function eventTitle(event) {
    const type = String(event?.event_type || event?.type || event?.name || "");
    const map = {
      page_view: "页面访问",
      upload: "图片上传",
      image_upload: "图片上传",
      segment_job: "分割任务",
      segment_complete: "分割完成",
      grounding: "Qwen 定位",
      qwen_request: "Qwen 定位",
      qwen_grounding: "Qwen 定位",
      agent_run: "Agent 流程启动",
      agent_complete: "Agent 流程完成",
    };
    return map[type] || (type ? type.replaceAll("_", " ") : "匿名事件");
  }

  function eventMeta(event) {
    const bits = [];
    const status = event?.status || event?.outcome;
    const durationMs = event?.duration_ms ?? event?.elapsed_ms;
    const timestamp = event?.created_at || event?.timestamp || event?.at;
    if (status) bits.push(String(status));
    if (durationMs !== undefined && durationMs !== null) bits.push(duration(durationMs));
    if (timestamp) bits.push(time(timestamp));
    return bits.length ? bits.join(" · ") : "无额外详情";
  }

  function renderEvents(events) {
    const safeEvents = Array.isArray(events) ? events.slice(0, 20) : [];
    elements.eventsList.replaceChildren();
    if (!safeEvents.length) {
      elements.eventsList.hidden = true;
      elements.eventsEmpty.hidden = false;
      elements.eventsEmpty.textContent = "这个统计窗口内还没有可显示的匿名事件。";
      return;
    }

    const fragment = document.createDocumentFragment();
    safeEvents.forEach((event) => {
      const item = document.createElement("li");
      item.className = "event-row";

      const icon = document.createElement("span");
      icon.className = "event-row__dot";
      icon.setAttribute("aria-hidden", "true");

      const copy = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = eventTitle(event);
      const meta = document.createElement("span");
      meta.textContent = eventMeta(event);
      copy.append(title, meta);

      item.append(icon, copy);
      fragment.append(item);
    });
    elements.eventsList.append(fragment);
    elements.eventsList.hidden = false;
    elements.eventsEmpty.hidden = true;
  }

  function render(data) {
    const summary = data?.summary && typeof data.summary === "object" ? data.summary : {};
    const timings = data?.timings && typeof data.timings === "object" ? data.timings : {};
    const actualWindow = number(data?.window_hours) || state.windowHours;

    // Prefer the current API schema, while tolerating the earlier field names so
    // an in-flight backend rollout does not leave the page unusable.
    metricElements.sessions.textContent = count(summary.active_sessions ?? summary.unique_sessions);
    metricElements.pageViews.textContent = count(summary.page_views);
    metricElements.uploads.textContent = count(summary.uploads ?? summary.image_uploads);
    metricElements.segmentJobs.textContent = count(summary.segment_jobs);
    metricElements.segmentNote.textContent = percent(summary.segment_success_rate);
    metricElements.qwen.textContent = count(summary.grounding_requests ?? summary.qwen_requests);
    metricElements.agent.textContent = count(summary.agent_runs);

    renderTiming(timingElements.qwen, timings.grounding_ms ?? timings.qwen);
    renderTiming(timingElements.sam2, timings.sam2_ms ?? timings.sam2);
    const coreTiming = timings.sam2_core_ms;
    const coreCount = number(coreTiming?.count);
    timingElements.sam2Core.card.hidden = !(coreTiming && coreCount && coreCount > 0);
    if (!timingElements.sam2Core.card.hidden) renderTiming(timingElements.sam2Core, coreTiming);
    renderTiming(timingElements.queue, timings.queue_wait_ms ?? timings.queue_wait);
    renderEvents(data?.recent_events);

    state.windowHours = actualWindow;
    refreshPeriodControls();
    elements.updatedAt.textContent = `更新于 ${time(data?.generated_at)}`;
  }

  function showAuthorizationMessage(message) {
    state.token = "";
    sessionStorage.removeItem(STORAGE_KEY);
    elements.tokenInput.value = "";
    elements.refreshButton.disabled = true;
    elements.authCard.classList.remove("auth-card--connected");
    elements.authMessage.textContent = message;
    elements.tokenInput.focus();
  }

  async function fetchMetrics() {
    if (!state.token || state.loading) return;

    setLoading(true);
    setNotice("");
    setConnection("loading", "正在读取数据");

    try {
      const url = new URL(endpoint, window.location.origin);
      url.searchParams.set("window_hours", String(state.windowHours));
      const response = await fetch(url, {
        headers: { "X-Ops-Token": state.token },
        cache: "no-store",
        credentials: "same-origin",
      });

      if (response.status === 401) {
        showAuthorizationMessage("令牌无效或已过期，请重新输入后再试。");
        setConnection("warning", "需要令牌");
        setNotice("无法读取运营数据：请检查运营访问令牌。", "warning");
        return;
      }
      if (response.status === 503) {
        setConnection("warning", "数据暂不可用");
        setNotice("运营数据服务暂时不可用。主站不受影响，稍后刷新即可。", "warning");
        return;
      }
      if (!response.ok) {
        throw new Error(`请求失败（${response.status}）`);
      }

      const data = await response.json();
      render(data);
      elements.authCard.classList.add("auth-card--connected");
      elements.authMessage.textContent = "已连接。本次会话结束后，令牌会自动清除。";
      setConnection("ready", "数据已连接");
    } catch (error) {
      setConnection("error", "连接失败");
      const detail = error instanceof Error && error.message ? `（${error.message}）` : "";
      setNotice(`无法读取运营数据，请检查网络或稍后重试${detail}`, "error");
    } finally {
      setLoading(false);
    }
  }

  function submitToken(event) {
    event.preventDefault();
    const token = elements.tokenInput.value.trim();
    if (!token) {
      elements.authMessage.textContent = "请输入运营访问令牌。";
      elements.tokenInput.focus();
      return;
    }
    state.token = token;
    sessionStorage.setItem(STORAGE_KEY, token);
    fetchMetrics();
  }

  function bindEvents() {
    elements.tokenForm.addEventListener("submit", submitToken);
    elements.refreshButton.addEventListener("click", fetchMetrics);
    elements.tokenVisibility.addEventListener("click", () => {
      const showing = elements.tokenInput.type === "text";
      elements.tokenInput.type = showing ? "password" : "text";
      elements.tokenVisibility.setAttribute("aria-pressed", String(!showing));
      elements.tokenVisibility.setAttribute("aria-label", showing ? "显示令牌" : "隐藏令牌");
      elements.tokenVisibility.title = showing ? "显示令牌" : "隐藏令牌";
    });
    elements.periodButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const hours = Number(button.dataset.windowHours);
        if (!Number.isFinite(hours) || hours <= 0 || state.windowHours === hours) return;
        state.windowHours = hours;
        refreshPeriodControls();
        if (state.token) fetchMetrics();
      });
    });
  }

  function initialize() {
    bindEvents();
    refreshPeriodControls();
    state.token = sessionStorage.getItem(STORAGE_KEY) || "";
    if (state.token) {
      elements.authMessage.textContent = "正在使用本次会话中保存的令牌连接…";
      elements.refreshButton.disabled = false;
      fetchMetrics();
    }
  }

  initialize();
})();
