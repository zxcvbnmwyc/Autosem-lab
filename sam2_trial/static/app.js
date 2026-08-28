(() => {
  "use strict";

  const fileInput = document.getElementById("file-input");
  const description = document.getElementById("description");
  const descriptionCount = document.getElementById("description-count");
  const fileName = document.getElementById("file-name");
  const canvas = document.getElementById("image-canvas");
  const context = canvas && canvas.getContext("2d");
  const canvasEmpty = document.getElementById("canvas-empty");
  const canvasShell = document.getElementById("canvas-shell");
  const status = document.getElementById("status");
  const modeHelp = document.getElementById("mode-help");
  const segmentButton = document.getElementById("segment-button");
  const groundButton = document.getElementById("ground-button");
  const runButton = document.getElementById("run-button");
  const groundingStatus = document.getElementById("grounding-status");
  const agentPanel = document.getElementById("agent-panel");
  const agentTitle = document.getElementById("agent-title");
  const agentStage = document.getElementById("agent-stage");
  const agentMessage = document.getElementById("agent-message");
  const agentActions = document.getElementById("agent-actions");
  const candidateChoices = document.getElementById("candidate-choices");
  const resetPromptsButton = document.getElementById("reset-prompts");
  const resultSection = document.getElementById("result-section");
  const resultPreview = document.getElementById("result-preview");
  const resultSummary = document.getElementById("result-summary");
  const maskLink = document.getElementById("mask-link");
  const contoursLink = document.getElementById("contours-link");
  const jsonLink = document.getElementById("json-link");
  const runtimeChip = document.getElementById("runtime-chip");
  const runtimeDetail = document.getElementById("runtime-detail");
  const jobCard = document.getElementById("job-card");
  const jobBadge = document.getElementById("job-badge");
  const jobElapsed = document.getElementById("job-elapsed");
  const jobTitleText = document.getElementById("job-title-text");
  const jobMessage = document.getElementById("job-message");
  const jobProgressBar = document.getElementById("job-progress-bar");
  const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));

  if (!fileInput || !description || !canvas || !context || !agentPanel || !agentTitle || !agentStage || !agentMessage || !agentActions) {
    return;
  }

  const activeJobStorageKey = "autosem:active-segmentation-job";
  const pollIntervalMs = 1500;
  const maxCanvasEdge = 1600;
  const state = {
    image: null,
    imageId: null,
    width: 0,
    height: 0,
    points: [],
    box: null,
    draftBox: null,
    boxDragStart: null,
    boxSource: null,
    groundingId: null,
    groundingCandidates: [],
    selectedCandidateIndex: null,
    agentRunId: null,
    agentPhase: null,
    agentMessage: "",
    agentEvaluation: null,
    groundingAvailable: false,
    mode: "positive",
    busy: false,
    activeJobId: null,
    pollUrl: null,
    pollTimer: null,
    pollFailures: 0,
    jobStartedAt: null,
    localPreviewUrl: null,
    previewToken: 0,
    redrawQueued: false,
  };

  const phaseCopy = {
    queued: "任务正在等待本地分割引擎。",
    loading_model: "正在加载 SAM2 模型。",
    encoding_image: "正在编码图片。",
    predicting: "SAM2 正在计算像素级边界。",
    rendering: "正在整理轮廓与下载文件。",
    succeeded: "轮廓已生成，可以下载结果。",
  };

  function setStatus(message, kind) {
    status.textContent = message;
    status.className = "status-message";
    if (kind) {
      status.classList.add("status-message--" + kind);
    }
  }

  function clamp(value, lower, upper) {
    return Math.min(Math.max(value, lower), upper);
  }

  function num(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function safeSessionGet(key) {
    try {
      return window.sessionStorage.getItem(key);
    } catch (_error) {
      return null;
    }
  }

  function safeSessionSet(key, value) {
    try {
      window.sessionStorage.setItem(key, value);
    } catch (_error) {
      // Browser storage is an optional convenience, never a requirement.
    }
  }

  function safeSessionRemove(key) {
    try {
      window.sessionStorage.removeItem(key);
    } catch (_error) {
      // Browser storage is an optional convenience, never a requirement.
    }
  }

  function resetJobPoll() {
    if (state.pollTimer) {
      window.clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function updateDescriptionCount() {
    descriptionCount.textContent = String(description.value.length) + " / 500";
  }

  function refreshActions() {
    const hasDescription = Boolean(description.value.trim());
    const hasPositivePoint = state.points.some((point) => point.label === 1);
    const hasManualPrompt = Boolean(state.box || hasPositivePoint);

    segmentButton.disabled = state.busy || !state.imageId || !hasManualPrompt;
    groundButton.disabled = state.busy || !state.imageId || !state.groundingAvailable || !hasDescription;
    runButton.disabled = state.busy || !state.imageId || !hasDescription;
  }

  function setBusy(isBusy, action) {
    state.busy = isBusy;
    fileInput.disabled = isBusy;
    description.disabled = isBusy;
    resetPromptsButton.disabled = isBusy;
    modeButtons.forEach((button) => {
      button.disabled = isBusy;
    });

    runButton.innerHTML = isBusy && action === "agent"
      ? "<span aria-hidden=\"true\">⋯</span> Agent 正在分析"
      : "<span aria-hidden=\"true\">✦</span> 启动分割 Agent";
    groundButton.textContent = isBusy && action === "ground" ? "Qwen 正在定位…" : "只查看 Qwen 候选";
    segmentButton.textContent = isBusy && action === "segment" ? "任务进行中…" : "生成轮廓";

    refreshActions();
    renderCandidateChoices();
    renderAgentPanel();
  }

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const displayX = clamp((event.clientX - rect.left) * canvas.width / Math.max(rect.width, 1), 0, canvas.width - 1);
    const displayY = clamp((event.clientY - rect.top) * canvas.height / Math.max(rect.height, 1), 0, canvas.height - 1);
    return {
      x: clamp(displayX * state.width / Math.max(canvas.width, 1), 0, state.width - 1),
      y: clamp(displayY * state.height / Math.max(canvas.height, 1), 0, state.height - 1),
    };
  }

  function toCanvasPoint(point) {
    return {
      x: point.x * canvas.width / Math.max(state.width, 1),
      y: point.y * canvas.height / Math.max(state.height, 1),
    };
  }

  function drawPoint(point) {
    const rect = canvas.getBoundingClientRect();
    const displayScale = canvas.width / Math.max(rect.width, 1);
    const canvasPoint = toCanvasPoint(point);
    const radius = Math.max(5 * displayScale, 4);
    const color = point.label === 1 ? "#27a66e" : "#d45656";
    context.beginPath();
    context.arc(canvasPoint.x, canvasPoint.y, radius, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
    context.lineWidth = Math.max(displayScale * 1.5, 1);
    context.strokeStyle = "#ffffff";
    context.stroke();
    context.fillStyle = "#ffffff";
    context.font = Math.max(9 * displayScale, 8) + "px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(point.label === 1 ? "+" : "−", canvasPoint.x, canvasPoint.y + displayScale * 0.5);
  }

  function drawBox(box, dashed) {
    if (!box) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const displayScale = canvas.width / Math.max(rect.width, 1);
    const start = toCanvasPoint({ x: box.x0, y: box.y0 });
    const end = toCanvasPoint({ x: box.x1, y: box.y1 });
    context.save();
    context.strokeStyle = "#e0ad2d";
    context.lineWidth = Math.max(displayScale * 2, 2);
    if (dashed) {
      context.setLineDash([5 * displayScale, 4 * displayScale]);
    }
    context.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
    context.restore();
  }

  function drawAgentCandidateBoxes() {
    if (state.agentPhase !== "needs_choice") {
      return;
    }
    const colors = ["#2b8a67", "#536fc8", "#b06527"];
    const rect = canvas.getBoundingClientRect();
    const displayScale = canvas.width / Math.max(rect.width, 1);
    state.groundingCandidates.forEach((candidate, index) => {
      const box = candidateToBox(candidate);
      if (!box) {
        return;
      }
      const start = toCanvasPoint({ x: box.x0, y: box.y0 });
      const end = toCanvasPoint({ x: box.x1, y: box.y1 });
      const color = colors[index % colors.length];
      context.save();
      context.setLineDash([6 * displayScale, 4 * displayScale]);
      context.lineWidth = Math.max(2 * displayScale, 2);
      context.strokeStyle = color;
      context.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
      context.setLineDash([]);
      const badge = Math.max(17 * displayScale, 15);
      context.fillStyle = color;
      context.fillRect(start.x, start.y, badge, badge);
      context.fillStyle = "#ffffff";
      context.font = Math.max(11 * displayScale, 10) + "px sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(index + 1), start.x + badge / 2, start.y + badge / 2 + displayScale * 0.5);
      context.restore();
    });
  }

  function redraw() {
    if (!state.image) {
      return;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(state.image, 0, 0, canvas.width, canvas.height);
    drawAgentCandidateBoxes();
    drawBox(state.box, false);
    drawBox(state.draftBox, true);
    state.points.forEach(drawPoint);
  }

  function scheduleRedraw() {
    if (state.redrawQueued) {
      return;
    }
    state.redrawQueued = true;
    window.requestAnimationFrame(() => {
      state.redrawQueued = false;
      redraw();
    });
  }

  function normalizedBox(start, end) {
    return {
      x0: Math.min(start.x, end.x),
      y0: Math.min(start.y, end.y),
      x1: Math.max(start.x, end.x),
      y1: Math.max(start.y, end.y),
    };
  }

  function clearAgentState() {
    state.agentRunId = null;
    state.agentPhase = null;
    state.agentMessage = "";
    state.agentEvaluation = null;
    agentActions.replaceChildren();
    agentPanel.hidden = true;
  }

  function agentPhaseMeta(phase) {
    return {
      needs_choice: { title: "需要你确认对象", stage: "待选择" },
      needs_confirmation: { title: "请确认候选框", stage: "待确认" },
      needs_manual_prompt: { title: "需要一个手动提示", stage: "等你标注" },
      ready_to_segment: { title: "准备交给 SAM2", stage: "可继续" },
      segmenting: { title: "SAM2 正在描边", stage: "处理中" },
      awaiting_evaluation: { title: "等待质量复核", stage: "待复核" },
      needs_refinement: { title: "建议再细化一次", stage: "可细化" },
      completed: { title: "已完成基础复核", stage: "完成" },
      failed: { title: "需要新的提示", stage: "可重试" },
    }[phase] || { title: "Agent 正在等待", stage: "待命" };
  }

  function addAgentAction(label, className, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button " + className;
    button.textContent = label;
    button.disabled = state.busy;
    button.addEventListener("click", handler);
    agentActions.append(button);
  }

  function renderAgentPanel() {
    if (!state.agentRunId || !state.agentPhase) {
      agentPanel.hidden = true;
      return;
    }
    const phase = state.agentPhase;
    const meta = agentPhaseMeta(phase);
    agentPanel.hidden = false;
    agentPanel.className = "agent-panel agent-panel--" + phase.replaceAll("_", "-");
    agentTitle.textContent = meta.title;
    agentStage.textContent = meta.stage;
    agentMessage.textContent = state.agentMessage || "Agent 正在整理下一步。";
    agentActions.replaceChildren();

    if (phase === "needs_choice") {
      addAgentAction("改用手动提示", "button--secondary", () => {
        clearAgentState();
        clearGrounding();
        setMode("positive");
        setStatus("已切换到手动提示。请在目标主体内添加一个包含点，或框选目标。", "");
      });
    } else if (phase === "needs_confirmation" || phase === "ready_to_segment") {
      addAgentAction("确认并交给 SAM2", "button--primary", startAgentSegmentation);
      addAgentAction("改用手动提示", "button--secondary", () => {
        clearAgentState();
        clearGrounding();
        setMode("positive");
        setStatus("已切换到手动提示。请添加点或框选一个目标区域。", "");
      });
    } else if (phase === "needs_manual_prompt") {
      addAgentAction("添加包含点", "button--primary", () => setMode("positive"));
      addAgentAction("框选目标", "button--secondary", () => setMode("box"));
    } else if (phase === "needs_refinement" || phase === "failed") {
      addAgentAction("添加包含点", "button--secondary", () => setMode("positive"));
      addAgentAction("框选细化", "button--secondary", () => setMode("box"));
      if (state.box || state.points.some((point) => point.label === 1)) {
        addAgentAction("使用新提示再试", "button--primary", startAgentSegmentation);
      }
    } else if (phase === "awaiting_evaluation") {
      addAgentAction("复核当前结果", "button--secondary", evaluateAgentResult);
    }
  }

  function applyAgentRun(run) {
    if (!run || typeof run.agent_id !== "string" || typeof run.phase !== "string") {
      throw new Error("Agent 没有返回有效的任务状态，请重新分析图片。" );
    }
    state.agentRunId = run.agent_id;
    state.agentPhase = run.phase;
    state.agentMessage = typeof run.message === "string" ? run.message : "Agent 正在等待下一步。";
    state.agentEvaluation = run.evaluation && typeof run.evaluation === "object" ? run.evaluation : null;
    state.groundingId = typeof run.grounding_id === "string" ? run.grounding_id : null;
    state.groundingCandidates = Array.isArray(run.candidates) ? run.candidates.slice(0, 3) : [];
    state.selectedCandidateIndex = Number.isInteger(run.selected_candidate_index) ? run.selected_candidate_index : null;
    if (state.selectedCandidateIndex !== null && state.groundingCandidates.length) {
      selectGroundingCandidate(state.selectedCandidateIndex);
    } else if (state.boxSource === "qwen") {
      clearQwenBox();
      redraw();
    }
    renderCandidateChoices();
    renderAgentPanel();
    refreshActions();
  }

  function clearGrounding() {
    state.groundingId = null;
    state.groundingCandidates = [];
    state.selectedCandidateIndex = null;
    candidateChoices.replaceChildren();
    candidateChoices.hidden = true;
  }

  function clearQwenBox() {
    if (state.boxSource === "qwen") {
      state.box = null;
      state.boxSource = null;
    }
  }

  function candidateToBox(candidate) {
    if (!candidate || !Array.isArray(candidate.box_xyxy) || candidate.box_xyxy.length !== 4) {
      return null;
    }
    const values = candidate.box_xyxy.map(num);
    if (values.some((value) => value === null)) {
      return null;
    }
    const box = normalizedBox(
      { x: clamp(values[0], 0, state.width - 1), y: clamp(values[1], 0, state.height - 1) },
      { x: clamp(values[2], 0, state.width - 1), y: clamp(values[3], 0, state.height - 1) }
    );
    return box.x1 - box.x0 >= 1 && box.y1 - box.y0 >= 1 ? box : null;
  }

  function selectGroundingCandidate(index) {
    const candidate = state.groundingCandidates[index];
    const box = candidateToBox(candidate);
    if (!box) {
      setStatus("Qwen 返回的候选框无效，请重新定位或手动框选。", "error");
      return false;
    }
    state.box = box;
    state.boxSource = "qwen";
    state.draftBox = null;
    state.boxDragStart = null;
    state.selectedCandidateIndex = index;
    redraw();
    renderCandidateChoices();
    refreshActions();
    return true;
  }

  function renderCandidateChoices() {
    candidateChoices.replaceChildren();
    if (!state.groundingCandidates.length) {
      candidateChoices.hidden = true;
      return;
    }

    const heading = document.createElement("p");
    heading.className = "candidate-heading";
    heading.textContent = state.agentPhase === "needs_choice"
      ? "Agent 找到了多个可能区域。请点选你真正想要的对象。"
      : state.agentPhase === "needs_confirmation"
        ? "Agent 找到了一个可能区域。请确认后再交给 SAM2。"
        : state.groundingCandidates.length === 1
      ? "Qwen 推荐了这个区域；可直接生成，也可再补充提示。"
      : "选择一个候选区域，再确认或细化边界。";
    candidateChoices.append(heading);

    const buttons = document.createElement("div");
    buttons.className = "candidate-buttons";
    state.groundingCandidates.forEach((candidate, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "candidate-button";
      const selected = index === state.selectedCandidateIndex;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
      button.disabled = state.busy;

      const confidence = num(candidate.confidence);
      const label = typeof candidate.label === "string" && candidate.label.trim()
        ? candidate.label.trim()
        : "候选区域";
      button.textContent = "候选 " + String(index + 1) + " · " + label + (confidence === null ? "" : " · " + String(Math.round(confidence * 100)) + "%");
      button.addEventListener("click", async () => {
        if (selectGroundingCandidate(index)) {
          setMode("box");
          if (state.agentRunId && state.agentPhase === "needs_choice") {
            await chooseAgentCandidate(index);
          } else {
            setStatus("已选择候选 " + String(index + 1) + "。可直接生成，或添加正负点细化。", "success");
          }
        }
      });
      buttons.append(button);
    });
    candidateChoices.append(buttons);
    candidateChoices.hidden = false;
  }

  function resetPrompts(options) {
    const settings = options || {};
    state.points = [];
    state.box = null;
    state.draftBox = null;
    state.boxDragStart = null;
    state.boxSource = null;
    clearGrounding();
    if (!settings.keepAgent) {
      clearAgentState();
    }
    redraw();
    refreshActions();
    if (!settings.quiet && state.imageId) {
      setStatus("提示已清空。请在图上添加包含点，或框选一个目标区域。", "");
    }
  }

  function setMode(mode) {
    state.mode = mode;
    modeButtons.forEach((button) => {
      const selected = button.dataset.mode === mode;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    if (mode === "positive") {
      modeHelp.textContent = "在目标主体内部点击，放置绿色包含点。";
    } else if (mode === "negative") {
      modeHelp.textContent = "在不应属于目标的区域点击，放置红色排除点。";
    } else {
      modeHelp.textContent = "围绕目标拖动一个框；框可以和正负点一起使用。";
    }
    canvas.style.cursor = mode === "box" ? "crosshair" : "copy";
  }

  async function readResponse(response) {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(typeof payload.error === "string" ? payload.error : "请求没有成功完成。请稍后再试。");
    }
    return payload;
  }

  function clearLocalPreview() {
    if (state.localPreviewUrl) {
      URL.revokeObjectURL(state.localPreviewUrl);
      state.localPreviewUrl = null;
    }
  }

  function displayCanvasImage(image, sourceWidth, sourceHeight) {
    const width = Math.max(2, Math.round(sourceWidth || image.naturalWidth));
    const height = Math.max(2, Math.round(sourceHeight || image.naturalHeight));
    const scale = Math.min(1, maxCanvasEdge / Math.max(width, height));
    state.image = image;
    state.width = width;
    state.height = height;
    canvas.width = Math.max(2, Math.round(width * scale));
    canvas.height = Math.max(2, Math.round(height * scale));
    canvasEmpty.hidden = true;
    canvas.hidden = false;
    canvasShell.classList.add("canvas-shell--has-image");
    redraw();
  }

  function beginLocalPreview(file, token) {
    clearLocalPreview();
    const objectUrl = URL.createObjectURL(file);
    state.localPreviewUrl = objectUrl;
    return new Promise((resolve) => {
      const image = new Image();
      image.onload = () => {
        const details = { image, width: image.naturalWidth, height: image.naturalHeight };
        if (token === state.previewToken) {
          displayCanvasImage(image, details.width, details.height);
          if (state.busy) {
            setStatus("图片预览已显示，正在上传…", "working");
          }
        }
        resolve(details);
      };
      image.onerror = () => resolve(null);
      image.src = objectUrl;
    });
  }

  function loadServerPreview(url, sourceWidth, sourceHeight, token) {
    if (typeof url !== "string" || !url) {
      return Promise.reject(new Error("服务器没有返回可显示的图片。"));
    }
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        if (token === state.previewToken) {
          clearLocalPreview();
          displayCanvasImage(image, sourceWidth, sourceHeight);
        }
        resolve();
      };
      image.onerror = () => reject(new Error("图片已经上传，但浏览器未能显示它。请换一张图片后重试。"));
      image.src = url;
    });
  }

  async function uploadImage(file) {
    resultSection.hidden = true;
    resetJobPoll();
    state.activeJobId = null;
    state.pollUrl = null;
    state.imageId = null;
    state.image = null;
    state.width = 0;
    state.height = 0;
    state.previewToken += 1;
    const previewToken = state.previewToken;
    safeSessionRemove(activeJobStorageKey);
    resetPrompts({ quiet: true });
    fileName.textContent = file.name;
    const form = new FormData();
    form.append("image", file);
    setBusy(true, "upload");
    setStatus("正在准备本地预览并上传图片…", "working");
    const localPreview = beginLocalPreview(file, previewToken);

    try {
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const payload = await readResponse(response);
      const local = await localPreview;
      if (previewToken !== state.previewToken) {
        return;
      }
      const sourceWidth = num(payload.width) || (local && local.width);
      const sourceHeight = num(payload.height) || (local && local.height);
      if (!sourceWidth || !sourceHeight || typeof payload.image_id !== "string") {
        throw new Error("服务器返回的图片信息不完整，请重新上传。" );
      }
      state.imageId = payload.image_id;
      const localMatchesServer = local && local.width === sourceWidth && local.height === sourceHeight;
      if (localMatchesServer) {
        state.width = sourceWidth;
        state.height = sourceHeight;
        redraw();
      } else {
        await loadServerPreview(payload.preview_url || payload.image_url, sourceWidth, sourceHeight, previewToken);
      }
      setBusy(false);
      setStatus("图片已载入。可以让 Qwen 自动定位，或直接在图上添加提示。", "success");
    } catch (error) {
      setBusy(false);
      setStatus(error.message, "error");
    }
  }

  async function autoGround() {
    if (!state.imageId) {
      setStatus("请先选择一张图片。", "error");
      return;
    }
    if (!description.value.trim()) {
      setStatus("请先写下你想定位的目标。", "error");
      return;
    }

    clearAgentState();
    clearGrounding();
    clearQwenBox();
    redraw();
    setBusy(true, "ground");
    setStatus("正在把图片和描述发送给已配置的视觉模型，生成候选区域…", "working");

    try {
      const response = await fetch("/api/ground", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: state.imageId, description: description.value.trim() }),
      });
      const result = await readResponse(response);
      const candidates = Array.isArray(result.candidates) ? result.candidates.slice(0, 3) : [];
      if (!candidates.length && result.candidate) {
        candidates.push(result.candidate);
      }
      if (!candidates.length) {
        setBusy(false);
        setStatus(typeof result.note === "string" ? result.note : "没有找到可用候选区域。可以直接手动点选或框选。", "error");
        return;
      }
      if (typeof result.grounding_id !== "string") {
        throw new Error("自动定位没有返回有效记录，请重新试一次。");
      }

      state.groundingId = result.grounding_id;
      state.groundingCandidates = candidates;
      if (!selectGroundingCandidate(0)) {
        setBusy(false);
        return;
      }
      setMode("box");
      setBusy(false);

      const note = typeof result.note === "string" && result.note ? " " + result.note : "";
      setStatus(candidates.length > 1
        ? "Qwen 返回了 " + String(candidates.length) + " 个候选区域，请选择最符合的一项。" + note
        : "Qwen 已返回候选区域。可直接生成，或添加正负点细化。" + note, "success");
    } catch (error) {
      setBusy(false);
      setStatus(error.message, "error");
    }
  }

  async function startAgent() {
    if (!state.imageId) {
      setStatus("请先选择一张图片。", "error");
      return;
    }
    if (!description.value.trim()) {
      setStatus("请先写下你想分割的目标。", "error");
      return;
    }

    clearGrounding();
    clearQwenBox();
    state.agentRunId = "pending";
    state.agentPhase = "locating";
    state.agentMessage = "我正在理解你的描述，并决定是直接分割、请你选择候选，还是请求一个手动提示。";
    state.agentEvaluation = null;
    redraw();
    renderAgentPanel();
    setBusy(true, "agent");
    setStatus("Agent 正在用 Qwen 理解目标…", "working");

    try {
      const response = await fetch("/api/agent-runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: state.imageId, description: description.value.trim() }),
      });
      const result = await readResponse(response);
      applyAgentRun(result);
      setBusy(false);
      setStatus(state.agentMessage, state.agentPhase === "needs_manual_prompt" ? "" : "success");
    } catch (error) {
      clearAgentState();
      setBusy(false);
      setStatus(error.message, "error");
    }
  }

  async function chooseAgentCandidate(index) {
    if (!state.agentRunId || state.agentRunId === "pending") {
      return;
    }
    setBusy(true, "agent");
    setStatus("Agent 正在确认你选择的候选区域…", "working");
    try {
      const response = await fetch("/api/agent-runs/" + encodeURIComponent(state.agentRunId) + "/choose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidate_index: index }),
      });
      const result = await readResponse(response);
      applyAgentRun(result);
      setBusy(false);
      setStatus(state.agentMessage, "success");
    } catch (error) {
      setBusy(false);
      setStatus(error.message, "error");
    }
  }

  function agentPromptPayload() {
    return {
      points: state.points.map((point) => ({ x: point.x, y: point.y, label: point.label })),
      box: state.box ? [state.box.x0, state.box.y0, state.box.x1, state.box.y1] : null,
    };
  }

  function startTrackingJob(job, fallbackMessage) {
    if (!job || typeof job.job_id !== "string" || !job.job_id) {
      throw new Error("任务没有返回编号，请重新提交。 ");
    }
    state.activeJobId = job.job_id;
    state.pollUrl = safePollUrl(job.poll_url, job.job_id);
    state.pollFailures = 0;
    state.jobStartedAt = Date.now();
    safeSessionSet(activeJobStorageKey, JSON.stringify({ id: state.activeJobId, pollUrl: state.pollUrl, startedAt: state.jobStartedAt }));
    updateJobCard({
      status: job.status || "queued",
      phase: job.phase || "queued",
      message: job.message || fallbackMessage || "任务已提交，正在等待本地引擎。",
    });
  }

  async function startAgentSegmentation() {
    if (!state.agentRunId || state.agentRunId === "pending") {
      return startSegmentation();
    }
    const hasPositivePoint = state.points.some((point) => point.label === 1);
    if (!state.box && !hasPositivePoint) {
      setStatus("请至少添加一个包含点或一个框选；排除点只能用于细化。", "error");
      return;
    }
    setBusy(true, "segment");
    setStatus("Agent 已确认提示，正在把任务交给 SAM2…", "working");
    updateJobCard({ status: "queued", phase: "queued", message: "Agent 正在进入本地分割队列。" });
    try {
      const response = await fetch("/api/agent-runs/" + encodeURIComponent(state.agentRunId) + "/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(agentPromptPayload()),
      });
      const result = await readResponse(response);
      applyAgentRun(result);
      startTrackingJob(result.job, "Agent 已提交任务，正在等待本地引擎。");
      await pollJob();
    } catch (error) {
      finishJobWithError(error.message);
    }
  }

  async function evaluateAgentResult() {
    if (!state.agentRunId || state.agentRunId === "pending") {
      return;
    }
    try {
      const response = await fetch("/api/agent-runs/" + encodeURIComponent(state.agentRunId) + "/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const result = await readResponse(response);
      applyAgentRun(result);
      if (state.agentPhase === "needs_refinement") {
        setStatus(state.agentMessage, "");
      }
    } catch (error) {
      // The finished SAM2 result remains useful even if the optional review failed.
      setStatus("轮廓已生成；Agent 暂时无法完成质量复核。", "");
    }
  }

  function buildSegmentationPayload() {
    return {
      image_id: state.imageId,
      description: description.value.trim(),
      points: state.points.map((point) => ({ x: point.x, y: point.y, label: point.label })),
      box: state.box ? [state.box.x0, state.box.y0, state.box.x1, state.box.y1] : null,
      grounding_id: state.groundingId,
      grounding_candidate_index: state.selectedCandidateIndex,
    };
  }

  function safePollUrl(value, jobId) {
    if (typeof value === "string" && value.startsWith("/")) {
      return value;
    }
    return "/api/jobs/" + encodeURIComponent(jobId);
  }

  async function startSegmentation() {
    if (!state.imageId) {
      setStatus("请先选择一张图片。", "error");
      return;
    }
    if (state.agentRunId && state.agentRunId !== "pending" && ["needs_confirmation", "needs_manual_prompt", "ready_to_segment", "needs_refinement"].includes(state.agentPhase)) {
      return startAgentSegmentation();
    }
    const hasPositivePoint = state.points.some((point) => point.label === 1);
    if (!state.box && !hasPositivePoint) {
      setStatus("请至少添加一个包含点或一个框选；排除点只能用于细化。", "error");
      return;
    }

    setBusy(true, "segment");
    setStatus("任务已提交。本地引擎会在后台完成分割，页面会持续更新状态。", "working");
    updateJobCard({ status: "queued", phase: "queued", message: "正在进入本地分割队列。" });

    try {
      const response = await fetch("/api/segment-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSegmentationPayload()),
      });
      const result = await readResponse(response);
      startTrackingJob(result, "任务已提交，正在等待本地引擎。");
      await pollJob();
    } catch (error) {
      finishJobWithError(error.message);
    }
  }

  function formatDuration(value) {
    const seconds = Math.max(0, Math.round(num(value) || 0));
    if (!seconds) {
      return "";
    }
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return minutes ? String(minutes) + " 分 " + String(remainder) + " 秒" : String(remainder) + " 秒";
  }

  function jobMeta(job) {
    const jobStatus = typeof job.status === "string" ? job.status.toLowerCase() : "queued";
    const phase = typeof job.phase === "string" ? job.phase.toLowerCase() : "";
    if (jobStatus === "succeeded") {
      return { badge: "已完成", title: "轮廓已经准备好", progress: 100, state: "success" };
    }
    if (jobStatus === "failed") {
      return { badge: "未完成", title: "这次任务没有成功", progress: 100, state: "error" };
    }
    if (jobStatus === "running") {
      const details = {
        loading_model: { title: "正在准备 SAM2 模型", progress: 30 },
        encoding_image: { title: "正在编码图片", progress: 48 },
        predicting: { title: "SAM2 正在推理边界", progress: 68 },
        rendering: { title: "正在整理结果文件", progress: 86 },
      };
      const current = details[phase] || { title: "任务正在运行", progress: 56 };
      return { badge: "运行中", title: current.title, progress: current.progress, state: "running" };
    }
    return { badge: "排队中", title: "等待本地分割引擎", progress: 18, state: "queued" };
  }

  function updateJobCard(job) {
    const meta = jobMeta(job);
    const phase = typeof job.phase === "string" ? job.phase.toLowerCase() : "";
    const message = typeof job.message === "string" && job.message.trim()
      ? job.message.trim()
      : phaseCopy[phase] || (meta.state === "queued" ? phaseCopy.queued : "正在更新任务状态。");
    const elapsed = num(job.elapsed_seconds);
    const localElapsed = state.jobStartedAt ? (Date.now() - state.jobStartedAt) / 1000 : 0;

    jobCard.className = "job-card job-card--" + meta.state;
    jobBadge.textContent = meta.badge;
    jobTitleText.textContent = meta.title;
    jobMessage.textContent = message;
    jobProgressBar.style.width = String(meta.progress) + "%";
    jobElapsed.textContent = formatDuration(elapsed === null ? localElapsed : elapsed);
  }

  function jobResult(job) {
    if (job && job.result && typeof job.result === "object") {
      return job.result;
    }
    return job || {};
  }

  function displayResult(rawResult) {
    const result = jobResult(rawResult);
    const previewUrl = typeof result.preview_url === "string" && result.preview_url
      ? result.preview_url
      : result.overlay_url;
    if (typeof previewUrl !== "string" || !previewUrl) {
      throw new Error("任务已经完成，但没有找到预览文件。请刷新后重试。 ");
    }
    resultPreview.src = previewUrl;
    const area = num(result.mask_area_px);
    const iou = num(result.estimated_iou);
    const pieces = [];
    if (area !== null) {
      pieces.push("已输出 " + Math.round(area).toLocaleString("zh-CN") + " 个 mask 像素");
    }
    if (iou !== null) {
      pieces.push("SAM2 估计 IoU：" + iou.toFixed(3));
    }
    resultSummary.textContent = pieces.length ? pieces.join("；") + "。" : "轮廓已经生成。";
    maskLink.href = typeof result.mask_url === "string" ? result.mask_url : "#";
    contoursLink.href = typeof result.contours_url === "string" ? result.contours_url : "#";
    jsonLink.href = typeof result.json_url === "string" ? result.json_url : "#";
    [maskLink, contoursLink, jsonLink].forEach((link) => {
      link.classList.toggle("is-disabled", link.href.endsWith("/#"));
      link.setAttribute("aria-disabled", String(link.href.endsWith("/#")));
    });
    resultSection.hidden = false;
  }

  function finishJobWithError(message) {
    resetJobPoll();
    state.activeJobId = null;
    state.pollUrl = null;
    state.jobStartedAt = null;
    safeSessionRemove(activeJobStorageKey);
    setBusy(false);
    updateJobCard({ status: "failed", message: message || "任务没有成功完成。" });
    setStatus(message || "任务没有成功完成。请调整提示后重试。", "error");
  }

  function finishJob(job) {
    resetJobPoll();
    const succeeded = job.status === "succeeded";
    state.activeJobId = null;
    state.pollUrl = null;
    state.jobStartedAt = null;
    safeSessionRemove(activeJobStorageKey);
    setBusy(false);
    updateJobCard(job);
    if (!succeeded) {
      if (state.agentRunId && state.agentRunId !== "pending") {
        void evaluateAgentResult();
      }
      finishJobWithError(typeof job.error === "string" ? job.error : job.message);
      return;
    }
    try {
      displayResult(job);
      setStatus("轮廓已经生成。可以下载 Mask、透明轮廓或 JSON。", "success");
      resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
      if (state.agentRunId && state.agentRunId !== "pending") {
        void evaluateAgentResult();
      }
    } catch (error) {
      setStatus(error.message, "error");
    }
  }

  async function pollJob() {
    if (!state.activeJobId || !state.pollUrl) {
      return;
    }
    try {
      const response = await fetch(state.pollUrl, { headers: { Accept: "application/json" }, cache: "no-store" });
      const job = await readResponse(response);
      state.pollFailures = 0;
      updateJobCard(job);

      if (job.status === "succeeded" || job.status === "failed") {
        finishJob(job);
        return;
      }
      state.pollTimer = window.setTimeout(pollJob, pollIntervalMs);
    } catch (error) {
      state.pollFailures += 1;
      if (state.pollFailures >= 3) {
        finishJobWithError("暂时无法读取任务状态。请刷新页面确认任务是否继续运行。 ");
        return;
      }
      updateJobCard({ status: "running", message: "正在重新连接任务状态…" });
      state.pollTimer = window.setTimeout(pollJob, pollIntervalMs * state.pollFailures);
    }
  }

  async function loadGroundingStatus() {
    try {
      const response = await fetch("/api/grounding/status", { cache: "no-store" });
      const result = await readResponse(response);
      state.groundingAvailable = Boolean(result.configured || result.ready);
      if (state.groundingAvailable) {
        groundingStatus.textContent = "自动定位已就绪" + (result.model ? " · " + result.model : "") + "。";
        groundingStatus.className = "engine-note engine-note--ready";
      } else {
        groundingStatus.textContent = "自动定位尚未配置；手动点选和框选仍可使用。";
        groundingStatus.className = "engine-note";
      }
    } catch (_error) {
      state.groundingAvailable = false;
      groundingStatus.textContent = "无法确认自动定位配置；手动点选和框选仍可使用。";
      groundingStatus.className = "engine-note";
    }
    refreshActions();
  }

  async function loadRuntimeStatus() {
    try {
      const response = await fetch("/api/runtime/status", { cache: "no-store" });
      const result = await readResponse(response);
      const ready = Boolean(result.ready);
      const device = typeof result.device === "string" ? result.device.toUpperCase() : "本地";
      const model = typeof result.model === "string" && result.model ? result.model : "SAM2";
      const depth = num(result.queue_depth);
      const capacity = num(result.queue_capacity);
      runtimeChip.className = "runtime-chip" + (ready ? " runtime-chip--ready" : " runtime-chip--warning");
      runtimeChip.lastElementChild.textContent = ready ? device + " 引擎可用" : "本地引擎未就绪";
      runtimeDetail.textContent = model + " · " + device + (depth !== null && capacity !== null ? " · 队列 " + String(depth) + "/" + String(capacity) : "");
    } catch (_error) {
      runtimeChip.className = "runtime-chip runtime-chip--warning";
      runtimeChip.lastElementChild.textContent = "无法读取本地引擎";
      runtimeDetail.textContent = "运行状态暂不可用";
    }
  }

  function restoreActiveJob() {
    const stored = safeSessionGet(activeJobStorageKey);
    if (!stored) {
      return;
    }
    try {
      const job = JSON.parse(stored);
      if (!job || typeof job.id !== "string") {
        safeSessionRemove(activeJobStorageKey);
        return;
      }
      state.activeJobId = job.id;
      state.pollUrl = safePollUrl(job.pollUrl, job.id);
      state.jobStartedAt = num(job.startedAt) || Date.now();
      setBusy(true, "segment");
      updateJobCard({ status: "running", message: "正在恢复任务状态…" });
      pollJob();
    } catch (_error) {
      safeSessionRemove(activeJobStorageKey);
    }
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) {
      uploadImage(file);
    }
  });

  description.addEventListener("input", () => {
    updateDescriptionCount();
    if (state.groundingId || state.agentRunId) {
      const clearsQwenBox = state.boxSource === "qwen";
      clearGrounding();
      clearAgentState();
      if (clearsQwenBox) {
        clearQwenBox();
        redraw();
      }
      setStatus("目标描述已修改，之前的 Agent 分析已失效；需要时请重新分析图片。", "");
    }
    refreshActions();
  });

  modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  resetPromptsButton.addEventListener("click", () => resetPrompts());
  groundButton.addEventListener("click", autoGround);
  runButton.addEventListener("click", startAgent);
  segmentButton.addEventListener("click", startSegmentation);

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.image || !state.imageId || state.busy) {
      return;
    }
    const point = pointFromEvent(event);
    if (state.mode === "box") {
      state.boxDragStart = point;
      state.draftBox = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
      canvas.setPointerCapture(event.pointerId);
    } else {
      if (state.agentRunId && state.agentPhase === "needs_choice") {
        clearAgentState();
        clearGrounding();
        setStatus("已切换到手动提示。包含点会直接交给 SAM2。", "");
      }
      state.points.push({ x: point.x, y: point.y, label: state.mode === "positive" ? 1 : 0 });
      refreshActions();
    }
    redraw();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.draftBox || !state.boxDragStart || state.busy || !state.imageId) {
      return;
    }
    state.draftBox = normalizedBox(state.boxDragStart, pointFromEvent(event));
    scheduleRedraw();
  });

  canvas.addEventListener("pointerup", (event) => {
    if (!state.draftBox || !state.boxDragStart || state.busy || !state.imageId) {
      return;
    }
    const completed = normalizedBox(state.boxDragStart, pointFromEvent(event));
    state.draftBox = null;
    state.boxDragStart = null;
    if (completed.x1 - completed.x0 >= 2 && completed.y1 - completed.y0 >= 2) {
      state.box = completed;
      state.boxSource = "manual";
      const keepsManualAgent = state.agentRunId && state.agentPhase === "needs_manual_prompt";
      if (!keepsManualAgent) {
        clearAgentState();
      }
      clearGrounding();
      setStatus("框选已添加。还可以添加正、负点来微调边界。", "success");
    } else {
      setStatus("框选太小，没有保存。", "error");
    }
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    redraw();
    refreshActions();
  });

  canvas.addEventListener("pointercancel", () => {
    state.draftBox = null;
    state.boxDragStart = null;
    redraw();
  });

  window.addEventListener("beforeunload", () => {
    resetJobPoll();
    clearLocalPreview();
  });

  setMode("positive");
  updateDescriptionCount();
  loadGroundingStatus();
  loadRuntimeStatus();
  restoreActiveJob();
})();
