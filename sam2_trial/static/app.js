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
  const processHub = document.getElementById("process-hub");
  const processEyebrow = document.getElementById("process-eyebrow");
  const processTitle = document.getElementById("process-title");
  const processStage = document.getElementById("process-stage");
  const processMessage = document.getElementById("process-message");
  const processSteps = Array.from(document.querySelectorAll("[data-process-step]"));
  const processPlan = document.getElementById("process-plan");
  const processPlanItems = document.getElementById("process-plan-items");
  const processPlanSource = document.getElementById("process-plan-source");
  const processPlanNote = document.getElementById("process-plan-note");
  const processActions = document.getElementById("process-actions");
  const processAssurance = document.getElementById("process-assurance");
  const modeHelp = document.getElementById("mode-help");
  const segmentButton = document.getElementById("segment-button");
  const groundButton = document.getElementById("ground-button");
  const runButton = document.getElementById("run-button");
  const oneClickButton = document.getElementById("one-click-button");
  const groundingStatus = document.getElementById("grounding-status");
  const candidateChoices = document.getElementById("candidate-choices");
  const resetPromptsButton = document.getElementById("reset-prompts");
  const resultSection = document.getElementById("result-section");
  const resultPreview = document.getElementById("result-preview");
  const resultSummary = document.getElementById("result-summary");
  const maskLink = document.getElementById("mask-link");
  const contoursLink = document.getElementById("contours-link");
  const jsonLink = document.getElementById("json-link");
  const runtimeChip = document.getElementById("runtime-chip");
  const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
  const undoPromptButton = document.getElementById("undo-prompt");
  const redoPromptButton = document.getElementById("redo-prompt");
  const toggleMaskOverlayButton = document.getElementById("toggle-mask-overlay");
  const showOriginalButton = document.getElementById("show-original-button");
  const editorSection = document.getElementById("editor-section");
  const editorNote = document.getElementById("editor-note");
  const maskAddButton = document.getElementById("mask-add-button");
  const maskEraseButton = document.getElementById("mask-erase-button");
  const maskBrushRadius = document.getElementById("mask-brush-radius");
  const maskBrushRadiusValue = document.getElementById("mask-brush-radius-value");
  const undoMaskStrokeButton = document.getElementById("undo-mask-stroke");
  const clearMaskStrokesButton = document.getElementById("clear-mask-strokes");
  const edgeOffset = document.getElementById("edge-offset");
  const edgeOffsetValue = document.getElementById("edge-offset-value");
  const featherPx = document.getElementById("feather-px");
  const featherPxValue = document.getElementById("feather-px-value");
  const maskCleanup = document.getElementById("mask-cleanup");
  const backgroundMode = document.getElementById("background-mode");
  const backgroundColorRow = document.getElementById("background-color-row");
  const backgroundColor = document.getElementById("background-color");
  const backgroundBlurRow = document.getElementById("background-blur-row");
  const backgroundBlurPx = document.getElementById("background-blur-px");
  const backgroundBlurPxValue = document.getElementById("background-blur-px-value");
  const backgroundBrightness = document.getElementById("background-brightness");
  const backgroundBrightnessValue = document.getElementById("background-brightness-value");
  const backgroundSaturation = document.getElementById("background-saturation");
  const backgroundSaturationValue = document.getElementById("background-saturation-value");
  const backgroundGrayscale = document.getElementById("background-grayscale");
  const subjectBrightness = document.getElementById("subject-brightness");
  const subjectBrightnessValue = document.getElementById("subject-brightness-value");
  const subjectContrast = document.getElementById("subject-contrast");
  const subjectContrastValue = document.getElementById("subject-contrast-value");
  const subjectSaturation = document.getElementById("subject-saturation");
  const subjectSaturationValue = document.getElementById("subject-saturation-value");
  const subjectHueDegrees = document.getElementById("subject-hue-degrees");
  const subjectHueDegreesValue = document.getElementById("subject-hue-degrees-value");
  const subjectTemperature = document.getElementById("subject-temperature");
  const subjectTemperatureValue = document.getElementById("subject-temperature-value");
  const subjectSharpen = document.getElementById("subject-sharpen");
  const subjectSharpenValue = document.getElementById("subject-sharpen-value");
  const subjectBlurPx = document.getElementById("subject-blur-px");
  const subjectBlurPxValue = document.getElementById("subject-blur-px-value");
  const subjectOpacity = document.getElementById("subject-opacity");
  const subjectOpacityValue = document.getElementById("subject-opacity-value");
  const outlineWidthPx = document.getElementById("outline-width-px");
  const outlineWidthPxValue = document.getElementById("outline-width-px-value");
  const outlineColor = document.getElementById("outline-color");
  const outlineOpacity = document.getElementById("outline-opacity");
  const outlineOpacityValue = document.getElementById("outline-opacity-value");
  const shadowOffsetX = document.getElementById("shadow-offset-x");
  const shadowOffsetXValue = document.getElementById("shadow-offset-x-value");
  const shadowOffsetY = document.getElementById("shadow-offset-y");
  const shadowOffsetYValue = document.getElementById("shadow-offset-y-value");
  const shadowBlurPx = document.getElementById("shadow-blur-px");
  const shadowBlurPxValue = document.getElementById("shadow-blur-px-value");
  const shadowColor = document.getElementById("shadow-color");
  const shadowOpacity = document.getElementById("shadow-opacity");
  const shadowOpacityValue = document.getElementById("shadow-opacity-value");
  const cropEnabled = document.getElementById("crop-enabled");
  const cropPaddingPx = document.getElementById("crop-padding-px");
  const cropPaddingPxValue = document.getElementById("crop-padding-px-value");
  const cropAspectRatio = document.getElementById("crop-aspect-ratio");
  const applyEditButton = document.getElementById("apply-edit");
  const resetEditButton = document.getElementById("reset-edit");
  const editResult = document.getElementById("edit-result");
  const editPreview = document.getElementById("edit-preview");
  const editSummary = document.getElementById("edit-summary");
  const editedDownloadLink = document.getElementById("edited-download-link");
  const editedMaskLink = document.getElementById("edited-mask-link");
  const editorFoldToggles = Array.from(document.querySelectorAll(".editor-fold__toggle"));
  const foldSummarySelection = document.getElementById("fold-summary-selection");
  const foldSummaryBackground = document.getElementById("fold-summary-background");
  const foldSummarySubject = document.getElementById("fold-summary-subject");
  const foldSummaryEffects = document.getElementById("fold-summary-effects");
  const foldSummaryCrop = document.getElementById("fold-summary-crop");
  const selectionInputs = [maskBrushRadius, edgeOffset, featherPx, maskCleanup].filter(Boolean);
  const backgroundInputs = [
    backgroundMode, backgroundColor, backgroundBlurPx, backgroundBrightness, backgroundSaturation, backgroundGrayscale,
  ].filter(Boolean);
  const subjectInputs = [
    subjectBrightness, subjectContrast, subjectSaturation, subjectHueDegrees,
    subjectTemperature, subjectSharpen, subjectBlurPx, subjectOpacity,
  ].filter(Boolean);
  const effectsInputs = [
    outlineWidthPx, outlineColor, outlineOpacity,
    shadowOffsetX, shadowOffsetY, shadowBlurPx, shadowColor, shadowOpacity,
  ].filter(Boolean);
  const cropInputs = [cropEnabled, cropPaddingPx, cropAspectRatio].filter(Boolean);
  const editorInputGroups = {
    selection: selectionInputs,
    background: backgroundInputs,
    subject: subjectInputs,
    effects: effectsInputs,
    crop: cropInputs,
  };
  const editorInputs = Object.values(editorInputGroups).flat();

  if (!fileInput || !description || !canvas || !context || !processHub || !processEyebrow || !processTitle || !processStage || !processMessage || !processPlan || !processPlanItems || !processPlanSource || !processPlanNote || !processActions || !processAssurance || !candidateChoices) {
    return;
  }

  const activeJobStorageKey = "autosem:active-segmentation-job";
  const pollIntervalMs = 1500;
  const maxCanvasEdge = 1600;
  const state = {
    image: null,
    baseImage: null,
    editImage: null,
    imageId: null,
    width: 0,
    height: 0,
    points: [],
    box: null,
    draftBox: null,
    boxDragStart: null,
    boxSource: null,
    promptUndo: [],
    promptRedo: [],
    resultId: null,
    selectionDirty: false,
    maskSource: null,
    maskOverlay: null,
    maskOverlayVisible: true,
    maskOverlayToken: 0,
    maskStrokes: [],
    activeMaskStroke: null,
    showingEdit: false,
    groundingId: null,
    groundingCandidates: [],
    selectedCandidateIndex: null,
    agentRunId: null,
    agentPhase: null,
    agentMessage: "",
    agentEvaluation: null,
    oneClickRunId: null,
    oneClickPhase: null,
    oneClickMessage: "",
    oneClickPlan: null,
    oneClickPlanNotice: "",
    oneClickCandidates: [],
    oneClickSelectedCandidateIndex: null,
    oneClickPollTimer: null,
    oneClickPollFailures: 0,
    groundingAvailable: false,
    mode: "positive",
    busy: false,
    busyAction: null,
    activeJobId: null,
    pollUrl: null,
    pollTimer: null,
    pollFailures: 0,
    jobStartedAt: null,
    processJob: null,
    noticeMessage: "上传图片后，说出要处理的主体；效果可以一起说。",
    noticeKind: "idle",
    localPreviewUrl: null,
    previewToken: 0,
    redrawQueued: false,
  };

  const phaseCopy = {
    queued: "等待处理。",
    loading_model: "正在准备。",
    encoding_image: "正在处理图片。",
    predicting: "正在生成选区。",
    rendering: "正在整理结果。",
    succeeded: "选区已生成。",
  };

  function setStatus(message, kind) {
    state.noticeMessage = typeof message === "string" && message.trim()
      ? message.trim()
      : "准备继续处理。";
    state.noticeKind = kind || "idle";
    renderProcessHub();
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
    if (oneClickButton) {
      oneClickButton.disabled = state.busy || !state.imageId || !state.groundingAvailable || !hasDescription;
    }
    if (undoPromptButton) {
      undoPromptButton.disabled = state.busy || !state.promptUndo.length;
    }
    if (redoPromptButton) {
      redoPromptButton.disabled = state.busy || !state.promptRedo.length;
    }
    refreshEditorControls();
  }

  function editorAvailable() {
    return Boolean(state.imageId && state.resultId && !state.selectionDirty);
  }

  function refreshEditorControls() {
    const available = editorAvailable();
    const disabled = state.busy || !available;
    if (applyEditButton) {
      applyEditButton.disabled = disabled;
    }
    if (resetEditButton) {
      resetEditButton.disabled = state.busy || !state.resultId;
    }
    editorInputs.forEach((input) => {
      input.disabled = disabled;
    });
    if (cropPaddingPx) cropPaddingPx.disabled = disabled || !(cropEnabled && cropEnabled.checked);
    if (cropAspectRatio) cropAspectRatio.disabled = disabled || !(cropEnabled && cropEnabled.checked);
    [maskAddButton, maskEraseButton, undoMaskStrokeButton, clearMaskStrokesButton].forEach((button) => {
      if (button) {
        button.disabled = disabled || !state.maskSource || (button === undoMaskStrokeButton && !state.maskStrokes.length) || (button === clearMaskStrokesButton && !state.maskStrokes.length);
      }
    });
    if (toggleMaskOverlayButton) {
      toggleMaskOverlayButton.hidden = !state.maskOverlay;
      toggleMaskOverlayButton.disabled = state.busy || !state.maskOverlay;
      toggleMaskOverlayButton.textContent = state.maskOverlayVisible ? "隐藏选区" : "显示选区";
    }
    if (showOriginalButton) {
      showOriginalButton.hidden = !state.editImage;
      showOriginalButton.disabled = state.busy || !state.editImage;
      showOriginalButton.textContent = state.showingEdit ? "查看原图" : "返回编辑预览";
    }
    if (editorNote && state.resultId) {
      editorNote.textContent = state.selectionDirty
        ? "提示已改，先更新选区。"
        : "改好后生成预览；每次都从原图重做。";
    }
  }

  function setBusy(isBusy, action) {
    state.busy = isBusy;
    state.busyAction = isBusy ? (action || null) : null;
    fileInput.disabled = isBusy;
    description.disabled = isBusy;
    resetPromptsButton.disabled = isBusy;
    modeButtons.forEach((button) => {
      button.disabled = isBusy;
    });

    runButton.innerHTML = isBusy && action === "agent"
      ? "<span aria-hidden=\"true\">⋯</span> 正在分析图片"
      : "<span aria-hidden=\"true\">✦</span> 自动选区";
    if (oneClickButton) {
      oneClickButton.innerHTML = isBusy && (action === "one-click" || action === "one-click-apply")
        ? "<span aria-hidden=\"true\">⋯</span> 正在一键处理"
        : "<span aria-hidden=\"true\">✦</span> 一键处理";
    }
    groundButton.textContent = isBusy && action === "ground" ? "正在找推荐位置…" : "查看推荐位置";
    segmentButton.textContent = isBusy && action === "segment"
      ? "任务进行中…"
      : state.selectionDirty ? "更新选区" : "生成选区";
    if (applyEditButton) {
      applyEditButton.innerHTML = isBusy && action === "edit"
        ? "<span aria-hidden=\"true\">⋯</span> 正在合成"
        : "生成编辑预览";
    }

    refreshActions();
    renderCandidateChoices();
    renderProcessHub();
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

  function drawMaskOverlay() {
    if (!state.maskOverlay || !state.maskOverlayVisible) {
      return;
    }
    context.save();
    context.globalAlpha = 0.42;
    context.drawImage(state.maskOverlay, 0, 0, canvas.width, canvas.height);
    context.restore();
  }

  function createMaskLayer(image) {
    const layer = document.createElement("canvas");
    layer.width = canvas.width;
    layer.height = canvas.height;
    const layerContext = layer.getContext("2d", { willReadFrequently: true });
    if (!layerContext) {
      return null;
    }
    layerContext.drawImage(image, 0, 0, layer.width, layer.height);
    const pixels = layerContext.getImageData(0, 0, layer.width, layer.height);
    for (let index = 0; index < pixels.data.length; index += 4) {
      const alpha = pixels.data[index];
      pixels.data[index] = 43;
      pixels.data[index + 1] = 166;
      pixels.data[index + 2] = 111;
      pixels.data[index + 3] = alpha;
    }
    layerContext.putImageData(pixels, 0, 0);
    return layer;
  }

  function cloneMaskLayer(layer) {
    const copy = document.createElement("canvas");
    copy.width = layer.width;
    copy.height = layer.height;
    const copyContext = copy.getContext("2d");
    if (copyContext) {
      copyContext.drawImage(layer, 0, 0);
    }
    return copy;
  }

  function drawMaskStrokeOnLayer(layer, stroke) {
    const layerContext = layer && layer.getContext("2d");
    if (!layerContext || !stroke || !Array.isArray(stroke.points) || !stroke.points.length) {
      return;
    }
    const scale = canvas.width / Math.max(state.width, 1);
    const radius = Math.max(1, Number(stroke.radius) * scale);
    const points = stroke.points.map(toCanvasPoint);
    layerContext.save();
    layerContext.globalCompositeOperation = stroke.mode === "erase" ? "destination-out" : "source-over";
    layerContext.strokeStyle = "#2ba66f";
    layerContext.fillStyle = "#2ba66f";
    layerContext.lineWidth = Math.max(1, radius * 2);
    layerContext.lineCap = "round";
    layerContext.lineJoin = "round";
    layerContext.beginPath();
    layerContext.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) {
      layerContext.lineTo(points[index].x, points[index].y);
    }
    layerContext.stroke();
    layerContext.beginPath();
    layerContext.arc(points[0].x, points[0].y, radius, 0, Math.PI * 2);
    layerContext.fill();
    if (points.length > 1) {
      const last = points[points.length - 1];
      layerContext.beginPath();
      layerContext.arc(last.x, last.y, radius, 0, Math.PI * 2);
      layerContext.fill();
    }
    layerContext.restore();
  }

  function rebuildMaskOverlay() {
    if (!state.maskSource) {
      state.maskOverlay = null;
      return;
    }
    const layer = cloneMaskLayer(state.maskSource);
    state.maskStrokes.forEach((stroke) => drawMaskStrokeOnLayer(layer, stroke));
    if (state.activeMaskStroke) {
      drawMaskStrokeOnLayer(layer, state.activeMaskStroke);
    }
    state.maskOverlay = layer;
  }

  function clearMaskOverlay() {
    state.maskOverlayToken += 1;
    state.maskSource = null;
    state.maskOverlay = null;
    state.maskStrokes = [];
    state.activeMaskStroke = null;
  }

  function loadMaskOverlay(url, options) {
    if (typeof url !== "string" || !url) {
      clearMaskOverlay();
      redraw();
      refreshActions();
      return;
    }
    const settings = options || {};
    const token = ++state.maskOverlayToken;
    const image = new Image();
    image.onload = () => {
      if (token !== state.maskOverlayToken) {
        return;
      }
      const layer = createMaskLayer(image);
      if (!layer) {
        return;
      }
      state.maskSource = layer;
      if (settings.resetStrokes) {
        state.maskStrokes = [];
      }
      rebuildMaskOverlay();
      redraw();
      refreshActions();
    };
    image.onerror = () => {
      if (token === state.maskOverlayToken) {
        setStatus("选区文件暂时无法加载，但仍可下载原始结果。", "error");
      }
    };
    image.src = url;
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
    if (state.showingEdit) {
      return;
    }
    drawMaskOverlay();
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

  function promptSnapshot() {
    return {
      points: state.points.map((point) => ({ x: point.x, y: point.y, label: point.label })),
      box: state.box ? { x0: state.box.x0, y0: state.box.y0, x1: state.box.x1, y1: state.box.y1 } : null,
      boxSource: state.boxSource,
    };
  }

  function rememberPromptSnapshot() {
    state.promptUndo.push(promptSnapshot());
    if (state.promptUndo.length > 40) {
      state.promptUndo.shift();
    }
    state.promptRedo = [];
  }

  function restorePromptSnapshot(snapshot) {
    state.points = Array.isArray(snapshot.points)
      ? snapshot.points.map((point) => ({ x: point.x, y: point.y, label: point.label }))
      : [];
    state.box = snapshot.box ? { x0: snapshot.box.x0, y0: snapshot.box.y0, x1: snapshot.box.x1, y1: snapshot.box.y1 } : null;
    state.boxSource = snapshot.boxSource || null;
    state.draftBox = null;
    state.boxDragStart = null;
    clearGrounding();
    clearAgentState();
  }

  function markSelectionDirty() {
    if (!state.resultId) {
      return;
    }
    state.selectionDirty = true;
    state.editImage = null;
    if (state.showingEdit && state.baseImage) {
      state.showingEdit = false;
      displayCanvasImage(state.baseImage, state.width, state.height, { asBase: false });
    } else {
      state.showingEdit = false;
    }
    if (editResult) {
      editResult.hidden = true;
    }
  }

  function undoPrompt() {
    if (!state.promptUndo.length || state.busy) {
      return;
    }
    state.promptRedo.push(promptSnapshot());
    restorePromptSnapshot(state.promptUndo.pop());
    markSelectionDirty();
    redraw();
    refreshActions();
    setStatus("已撤销上一步提示。更新选区后即可继续编辑。", "");
  }

  function redoPrompt() {
    if (!state.promptRedo.length || state.busy) {
      return;
    }
    state.promptUndo.push(promptSnapshot());
    restorePromptSnapshot(state.promptRedo.pop());
    markSelectionDirty();
    redraw();
    refreshActions();
    setStatus("已恢复提示。更新选区后即可继续编辑。", "");
  }

  function setMaskTool(mode) {
    if (!editorAvailable()) {
      return;
    }
    state.mode = mode;
    modeButtons.forEach((button) => {
      button.classList.remove("selected");
      button.setAttribute("aria-pressed", "false");
    });
    if (maskAddButton) {
      const selected = mode === "mask-add";
      maskAddButton.classList.toggle("selected", selected);
      maskAddButton.setAttribute("aria-pressed", String(selected));
    }
    if (maskEraseButton) {
      const selected = mode === "mask-erase";
      maskEraseButton.classList.toggle("selected", selected);
      maskEraseButton.setAttribute("aria-pressed", String(selected));
    }
    modeHelp.textContent = mode === "mask-add"
      ? "在画布上拖动，补进应属于选区的区域。"
      : "在画布上拖动，擦除不应属于选区的区域。";
    canvas.style.cursor = "crosshair";
  }

  function resetMaskToolButtons() {
    [maskAddButton, maskEraseButton].forEach((button) => {
      if (button) {
        button.classList.remove("selected");
        button.setAttribute("aria-pressed", "false");
      }
    });
  }

  function beginMaskStroke(point, pointerId) {
    const radius = Math.max(1, Number(maskBrushRadius && maskBrushRadius.value) || 24);
    state.activeMaskStroke = {
      mode: state.mode === "mask-erase" ? "erase" : "add",
      radius,
      points: [point],
    };
    canvas.setPointerCapture(pointerId);
    rebuildMaskOverlay();
    redraw();
  }

  function extendMaskStroke(point) {
    if (!state.activeMaskStroke) {
      return;
    }
    const points = state.activeMaskStroke.points;
    const previous = points[points.length - 1];
    if (Math.hypot(point.x - previous.x, point.y - previous.y) < 0.75) {
      return;
    }
    points.push(point);
    rebuildMaskOverlay();
    scheduleRedraw();
  }

  function completeMaskStroke(pointerId) {
    if (!state.activeMaskStroke) {
      return false;
    }
    state.maskStrokes.push(state.activeMaskStroke);
    state.activeMaskStroke = null;
    if (canvas.hasPointerCapture(pointerId)) {
      canvas.releasePointerCapture(pointerId);
    }
    rebuildMaskOverlay();
    redraw();
    refreshActions();
    setStatus("已记录笔刷，可生成原图尺寸预览。", "success");
    return true;
  }

  function undoMaskStroke() {
    if (!state.maskStrokes.length || state.busy) {
      return;
    }
    state.maskStrokes.pop();
    rebuildMaskOverlay();
    redraw();
    refreshActions();
    setStatus("已撤销最后一笔选区修正。", "");
  }

  function clearMaskStrokes() {
    if (!state.maskStrokes.length || state.busy) {
      return;
    }
    state.maskStrokes = [];
    rebuildMaskOverlay();
    redraw();
    refreshActions();
    setStatus("手动选区笔刷已清除。", "");
  }

  function clearAgentState() {
    state.agentRunId = null;
    state.agentPhase = null;
    state.agentMessage = "";
    state.agentEvaluation = null;
    renderCandidateChoices();
    renderProcessHub();
  }

  function resetOneClickPoll() {
    if (state.oneClickPollTimer) {
      window.clearTimeout(state.oneClickPollTimer);
      state.oneClickPollTimer = null;
    }
  }

  function clearOneClickState() {
    resetOneClickPoll();
    state.oneClickRunId = null;
    state.oneClickPhase = null;
    state.oneClickMessage = "";
    state.oneClickPlan = null;
    state.oneClickPlanNotice = "";
    state.oneClickCandidates = [];
    state.oneClickSelectedCandidateIndex = null;
    state.oneClickPollFailures = 0;
    renderCandidateChoices();
    renderProcessHub();
  }

  function oneClickPhaseMeta(phase) {
    return {
      planning: { title: "正在理解你的需求", stage: "第 1 步", step: 1, tone: "working" },
      needs_target_confirmation: { title: "请确认要处理的对象", stage: "待确认", step: 2, tone: "attention" },
      segmenting: { title: "正在生成选区", stage: "第 3 步", step: 3, tone: "working" },
      selection_ready: { title: "选区已准备好", stage: "可检查", step: 4, tone: "success" },
      ready_to_apply: { title: "请确认这次处理", stage: "方案待确认", step: 4, tone: "attention" },
      composing: { title: "正在生成编辑预览", stage: "第 5 步", step: 5, tone: "working" },
      completed: { title: "预览已生成", stage: "完成", step: 5, tone: "success" },
      needs_input: { title: "还差一步", stage: "需要补充", step: 1, tone: "attention" },
      unsupported: { title: "暂不能这样处理", stage: "当前受限", step: 1, tone: "error" },
      failed: { title: "处理暂未完成", stage: "可重试", step: 1, tone: "error" },
    }[phase] || { title: "正在处理", stage: "处理中", step: 1, tone: "working" };
  }

  function planInteger(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.round(parsed) : fallback;
  }

  function validHexColor(value, fallback) {
    return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value)
      ? value.toUpperCase()
      : fallback;
  }

  function readPlanSection(plan, key) {
    return plan && plan[key] && typeof plan[key] === "object" ? plan[key] : {};
  }

  function isSmallViewport() {
    return window.matchMedia("(max-width: 560px)").matches;
  }

  function foldElements(name) {
    const root = document.querySelector('[data-editor-fold="' + name + '"]');
    if (!root) {
      return null;
    }
    const toggle = root.querySelector(".editor-fold__toggle");
    const bodyId = toggle && toggle.getAttribute("aria-controls");
    const body = bodyId ? document.getElementById(bodyId) : null;
    return toggle && body ? { root, toggle, body } : null;
  }

  function setFoldExpanded(name, expanded, options) {
    const elements = foldElements(name);
    if (!elements) {
      return;
    }
    const settings = options || {};
    const shouldExpand = Boolean(expanded);
    elements.root.classList.toggle("is-collapsed", !shouldExpand);
    elements.root.classList.toggle("is-suggested", Boolean(settings.suggested));
    elements.toggle.setAttribute("aria-expanded", String(shouldExpand));
    elements.body.hidden = !shouldExpand;
    if (shouldExpand && isSmallViewport() && !settings.skipAccordion) {
      Object.keys(editorInputGroups).forEach((groupName) => {
        if (groupName !== name) {
          setFoldExpanded(groupName, false, { skipAccordion: true });
        }
      });
    }
  }

  function pulseSuggestedFold(name) {
    const elements = foldElements(name);
    if (!elements) {
      return;
    }
    elements.root.classList.add("is-suggested");
    window.setTimeout(() => {
      elements.root.classList.remove("is-suggested");
    }, 1800);
  }

  function expandPlanDrivenFolds(plan) {
    const selection = readPlanSection(plan, "selection");
    const background = readPlanSection(plan, "background");
    const subject = readPlanSection(plan, "subject");
    const effects = readPlanSection(plan, "effects");
    const crop = readPlanSection(plan, "crop");
    const activeGroups = [];
    if ((selection.cleanup === false) || planInteger(selection.edge_offset, 0) || planInteger(selection.feather_px, 0)) activeGroups.push("selection");
    if ((background.mode && background.mode !== "original") || planInteger(background.blur_px, 18) !== 18 || planInteger(background.brightness, 0) || planInteger(background.saturation, 0) || Boolean(background.grayscale)) activeGroups.push("background");
    if (planInteger(subject.brightness, 0) || planInteger(subject.contrast, 0) || planInteger(subject.saturation, 0) || planInteger(subject.hue_degrees, 0) || planInteger(subject.temperature, 0) || planInteger(subject.sharpen, 0) || planInteger(subject.blur_px, 0) || planInteger(subject.opacity, 100) !== 100) activeGroups.push("subject");
    if (planInteger(effects.outline_width_px, 0) || planInteger(effects.outline_opacity, 100) !== 100 || planInteger(effects.shadow_offset_x, 0) || planInteger(effects.shadow_offset_y, 0) || planInteger(effects.shadow_blur_px, 0) || planInteger(effects.shadow_opacity, 0)) activeGroups.push("effects");
    if (Boolean(crop.enabled) || planInteger(crop.padding_px, 24) !== 24 || (typeof crop.aspect_ratio === "string" && crop.aspect_ratio !== "free")) activeGroups.push("crop");
    if (!activeGroups.length) {
      activeGroups.push("selection", "background");
    }
    activeGroups.forEach((name, index) => {
      setFoldExpanded(name, true, { suggested: index === 0 });
      pulseSuggestedFold(name);
    });
  }

  function addPlanItem(label, value, swatch) {
    const item = document.createElement("div");
    item.className = "process-plan-item";
    if (swatch) {
      const swatchElement = document.createElement("i");
      swatchElement.className = "process-plan-swatch" + (swatch.className ? " " + swatch.className : "");
      if (swatch.color) swatchElement.style.setProperty("--swatch", swatch.color);
      swatchElement.setAttribute("aria-hidden", "true");
      item.append(swatchElement);
    }
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = value;
    item.append(labelElement, valueElement);
    processPlanItems.append(item);
  }

  function renderProcessPlan(plan, planNotice) {
    processPlanItems.replaceChildren();
    const target = plan && typeof plan.target === "string" ? plan.target.trim().slice(0, 80) : "";
    if (!target) {
      processPlan.hidden = true;
      processPlanNote.textContent = typeof planNotice === "string" ? planNotice : "";
      return;
    }

    const reasonCode = plan && typeof plan.reason_code === "string" ? plan.reason_code : "";
    if (reasonCode === "selection_only") {
      addPlanItem("处理主体", target);
      addPlanItem("处理", "只生成选区");
      processPlanSource.textContent = "已校验";
      processPlanNote.textContent = typeof planNotice === "string" && planNotice.trim()
        ? planNotice.trim()
        : "这次只会生成可检查、可微调的选区，不会直接套用编辑效果。";
      processPlan.hidden = false;
      return;
    }

    const selection = readPlanSection(plan, "selection");
    const background = readPlanSection(plan, "background");
    const subject = readPlanSection(plan, "subject");
    const effects = readPlanSection(plan, "effects");
    const crop = readPlanSection(plan, "crop");
    addPlanItem("处理主体", target);

    const backgroundMode = typeof background.mode === "string" ? background.mode : "original";
    const backgroundAdjustments = [];
    const backgroundBrightnessValue = planInteger(background.brightness, 0);
    const backgroundSaturationValue = planInteger(background.saturation, 0);
    if (backgroundBrightnessValue) backgroundAdjustments.push("亮度 " + signedValue(backgroundBrightnessValue));
    if (backgroundSaturationValue) backgroundAdjustments.push("饱和度 " + signedValue(backgroundSaturationValue));
    if (background.grayscale) backgroundAdjustments.push("灰度");
    if (backgroundMode === "transparent") {
      addPlanItem("背景", ["透明背景"].concat(backgroundAdjustments).join(" · "), { className: "process-plan-swatch--transparent" });
    } else if (backgroundMode === "color") {
      const color = validHexColor(background.color, "#FFFFFF");
      addPlanItem("背景", [color === "#FFFFFF" ? "白色背景" : "纯色背景 " + color].concat(backgroundAdjustments).join(" · "), { color });
    } else if (backgroundMode === "blur") {
      const blur = planInteger(background.blur_px, 18);
      addPlanItem("背景", ["虚化 " + String(blur) + " px"].concat(backgroundAdjustments).join(" · "), { className: "process-plan-swatch--blur" });
    } else {
      addPlanItem("背景", backgroundAdjustments.length ? "保留原背景 · " + backgroundAdjustments.join(" · ") : "保留原背景");
    }

    const subjectAdjustments = [];
    const brightness = planInteger(subject.brightness, 0);
    const contrast = planInteger(subject.contrast, 0);
    const saturation = planInteger(subject.saturation, 0);
    const hueDegrees = planInteger(subject.hue_degrees, 0);
    const temperature = planInteger(subject.temperature, 0);
    const sharpen = planInteger(subject.sharpen, 0);
    const blur = planInteger(subject.blur_px, 0);
    const opacity = planInteger(subject.opacity, 100);
    if (brightness) subjectAdjustments.push("亮度 " + (brightness > 0 ? "+" : "") + String(brightness));
    if (contrast) subjectAdjustments.push("对比度 " + signedValue(contrast));
    if (saturation) subjectAdjustments.push("饱和度 " + (saturation > 0 ? "+" : "") + String(saturation));
    if (hueDegrees) subjectAdjustments.push("色相 " + signedValue(hueDegrees) + "°");
    if (temperature) subjectAdjustments.push("冷暖 " + signedValue(temperature));
    if (sharpen) subjectAdjustments.push("锐化 " + String(sharpen));
    if (blur) subjectAdjustments.push("柔化 " + String(blur) + " px");
    if (opacity !== 100) subjectAdjustments.push("不透明度 " + String(opacity) + "%");
    addPlanItem("主体", subjectAdjustments.length ? subjectAdjustments.join(" · ") : "保持原样");

    const edgeAdjustments = [];
    const edgeOffset = planInteger(selection.edge_offset, 0);
    const feather = planInteger(selection.feather_px, 0);
    if (selection.cleanup !== false) edgeAdjustments.push("自动清理");
    if (edgeOffset) edgeAdjustments.push(edgeOffset > 0 ? "边缘扩展 " + String(edgeOffset) + " px" : "边缘收缩 " + String(Math.abs(edgeOffset)) + " px");
    if (feather) edgeAdjustments.push("羽化 " + String(feather) + " px");
    addPlanItem("边缘", edgeAdjustments.length ? edgeAdjustments.join(" · ") : "保持原样");

    const effectAdjustments = [];
    const outlineWidth = planInteger(effects.outline_width_px, 0);
    const outlineOpacityValue = planInteger(effects.outline_opacity, 100);
    const shadowX = planInteger(effects.shadow_offset_x, 0);
    const shadowY = planInteger(effects.shadow_offset_y, 0);
    const shadowBlurValue = planInteger(effects.shadow_blur_px, 0);
    const shadowOpacityValue = planInteger(effects.shadow_opacity, 0);
    if (outlineWidth) effectAdjustments.push("描边 " + String(outlineWidth) + " px");
    if (outlineWidth && outlineOpacityValue !== 100) effectAdjustments.push("描边透明度 " + String(outlineOpacityValue) + "%");
    if (shadowX || shadowY || shadowBlurValue || shadowOpacityValue) {
      effectAdjustments.push("阴影 " + String(shadowX) + "," + String(shadowY));
      if (shadowBlurValue) effectAdjustments.push("模糊 " + String(shadowBlurValue) + " px");
      if (shadowOpacityValue) effectAdjustments.push("透明度 " + String(shadowOpacityValue) + "%");
    }
    addPlanItem("样式", effectAdjustments.length ? effectAdjustments.join(" · ") : "无额外样式");

    const cropEnabledValue = Boolean(crop.enabled);
    const cropPaddingValue = planInteger(crop.padding_px, 24);
    const cropRatioValue = typeof crop.aspect_ratio === "string" ? crop.aspect_ratio : "free";
    addPlanItem("画布", cropEnabledValue
      ? "裁切 · 留白 " + String(cropPaddingValue) + " px · 比例 " + cropRatioValue
      : "不裁切");

    processPlanSource.textContent = "已校验";
    processPlanNote.textContent = typeof planNotice === "string" && planNotice.trim()
      ? planNotice.trim()
      : "展示的是服务端已校验的执行项；不会删除、补全、换物或改动其他主体。";
    processPlan.hidden = false;
  }

  function addProcessAction(label, className, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button " + className;
    button.textContent = label;
    button.disabled = state.busy;
    button.addEventListener("click", () => { void handler(); });
    processActions.append(button);
  }

  function addProcessDownload(link, label) {
    if (!link || typeof link.href !== "string" || !link.href || link.href.endsWith("/#")) {
      return;
    }
    const anchor = document.createElement("a");
    anchor.className = "button button--primary";
    anchor.href = link.href;
    anchor.download = "";
    anchor.textContent = label;
    processActions.append(anchor);
  }

  function focusDescription(message) {
    if (state.busy) return;
    description.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => description.focus(), 260);
    if (message) setStatus(message, "");
  }

  function switchToManualSelection() {
    if (state.busy) return;
    clearOneClickState();
    clearAgentState();
    clearGrounding();
    clearQwenBox();
    redraw();
    setMode("positive");
    setStatus("已切换到手动选区。请点选或框选主体。", "");
  }

  function openEditor() {
    if (editorSection && !editorSection.hidden) {
      editorSection.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    canvasShell.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function oneClickProcessPresentation() {
    const phase = state.oneClickPhase;
    const meta = oneClickPhaseMeta(phase);
    const target = state.oneClickPlan && typeof state.oneClickPlan.target === "string"
      ? state.oneClickPlan.target.trim()
      : "";
    let message = state.oneClickMessage || "正在准备处理。";
    if (phase === "selection_ready") {
      message = target
        ? "已找到「" + target + "」并生成选区。现在可以直接检查或继续微调。"
        : "选区已生成。现在可以直接检查或继续微调。";
    } else if (phase === "ready_to_apply") {
      message = target
        ? "已找到「" + target + "」并生成选区。确认后才会生成编辑预览。"
        : "选区已生成。确认后才会生成编辑预览。";
    } else if (phase === "segmenting" && target) {
      message = "已定位「" + target + "」，正在勾出它的边界。";
    } else if (phase === "composing") {
      message = "正在按已确认的方案合成，原图不会被覆盖。";
    }
    return {
      source: "one-click",
      phase,
      eyebrow: "一键处理",
      title: meta.title,
      stage: meta.stage,
      step: meta.step,
      tone: meta.tone,
      message,
      plan: state.oneClickPlan,
      planNotice: state.oneClickPlanNotice,
      assurance: phase === "ready_to_apply"
        ? "确认前不会修改图片；原图始终保留。"
        : phase === "selection_ready"
          ? "这一步只生成选区；原图始终保留，之后仍可继续微调。"
          : "原图不会被覆盖；生成的是可继续调整的预览。",
    };
  }

  function agentPhaseMeta(phase) {
    return {
      needs_choice: { title: "请确认要选的对象", stage: "待选择", step: 2, tone: "attention" },
      needs_confirmation: { title: "请确认候选对象", stage: "待确认", step: 2, tone: "attention" },
      needs_manual_prompt: { title: "需要一个手动提示", stage: "等待标注", step: 2, tone: "attention" },
      ready_to_segment: { title: "准备生成选区", stage: "待确认", step: 3, tone: "attention" },
      segmenting: { title: "正在生成选区", stage: "第 3 步", step: 3, tone: "working" },
      awaiting_evaluation: { title: "正在检查选区", stage: "检查中", step: 3, tone: "working" },
      needs_refinement: { title: "建议再细化一次", stage: "需要调整", step: 3, tone: "attention" },
      completed: { title: "选区已生成", stage: "完成", step: 3, tone: "success" },
      failed: { title: "需要新的提示", stage: "可重试", step: 3, tone: "error" },
    }[phase] || { title: "正在准备下一步", stage: "待命", step: 2, tone: "working" };
  }

  function agentProcessPresentation() {
    const meta = agentPhaseMeta(state.agentPhase);
    return {
      source: "agent",
      phase: state.agentPhase,
      eyebrow: "自动选区",
      title: meta.title,
      stage: meta.stage,
      step: meta.step,
      tone: meta.tone,
      message: state.agentMessage || "正在准备下一步。",
      plan: null,
      assurance: "可以随时改用点选、框选和笔刷细化选区。",
    };
  }

  function jobProcessPresentation() {
    const job = state.processJob || {};
    const meta = jobMeta(job);
    const phase = typeof job.phase === "string" ? job.phase.toLowerCase() : "";
    let message = typeof job.message === "string" && job.message.trim()
      ? job.message.trim()
      : phaseCopy[phase] || "正在更新处理状态。";
    const elapsed = num(job.elapsed_seconds);
    const localElapsed = state.jobStartedAt ? (Date.now() - state.jobStartedAt) / 1000 : 0;
    const elapsedText = formatDuration(elapsed === null ? localElapsed : elapsed);
    if (elapsedText && meta.state !== "success") message += " 已用 " + elapsedText + "。";
    return {
      source: "job",
      phase: meta.state,
      eyebrow: "选区处理",
      title: meta.title,
      stage: meta.state === "success" ? "完成" : meta.state === "error" ? "可重试" : "第 3 步",
      step: 3,
      tone: meta.state === "success" ? "success" : meta.state === "error" ? "error" : "working",
      message,
      plan: null,
      assurance: "SAM2 只负责勾出选区；之后仍可手动修边。",
    };
  }

  function noticeProcessPresentation() {
    const kind = state.noticeKind || "idle";
    const hasImage = Boolean(state.imageId);
    const tone = kind === "error" ? "error" : kind === "working" ? "working" : kind === "success" ? "success" : "idle";
    return {
      source: "notice",
      phase: kind,
      eyebrow: hasImage ? "工作台" : "智能剪辑",
      title: kind === "working" ? "正在准备" : hasImage ? "图片已准备好" : "准备开始",
      stage: kind === "error" ? "需要处理" : hasImage ? "等待描述" : "等待图片",
      step: kind === "working" && hasImage ? 1 : 0,
      tone,
      message: state.noticeMessage || "上传图片后，说出要处理的主体；效果可以一起说。",
      plan: null,
      assurance: "一次处理一个画面中可见的主体；原图始终保留。",
    };
  }

  function currentProcessPresentation() {
    if (state.busy && ["upload", "edit"].includes(state.busyAction)) return noticeProcessPresentation();
    if (state.oneClickRunId && state.oneClickPhase) return oneClickProcessPresentation();
    const jobStatus = state.processJob && typeof state.processJob.status === "string"
      ? state.processJob.status.toLowerCase()
      : "";
    if (state.processJob && !["succeeded", "failed"].includes(jobStatus)) return jobProcessPresentation();
    if (state.busy && state.noticeKind === "working") return noticeProcessPresentation();
    if (state.agentRunId && state.agentPhase) return agentProcessPresentation();
    if (state.processJob) return jobProcessPresentation();
    return noticeProcessPresentation();
  }

  function renderProcessActions(model) {
    processActions.replaceChildren();
    if (model.source === "one-click") {
      if (model.phase === "ready_to_apply") {
        addProcessAction("确认并生成预览", "button--primary", confirmOneClickPlan);
        addProcessAction("修改需求", "button--secondary", () => focusDescription("可以修改描述后重新处理。"));
      } else if (model.phase === "selection_ready") {
        addProcessAction("继续微调", "button--primary", openEditor);
        addProcessAction("修改需求", "button--secondary", () => focusDescription("如果还想换背景、调色或加样式，可以继续补充。"));
      } else if (model.phase === "needs_target_confirmation") {
        addProcessAction("修改需求", "button--secondary", () => focusDescription("可以补充位置或外观后重新处理。"));
        addProcessAction("改用手动选区", "button--secondary", switchToManualSelection);
      } else if (model.phase === "needs_input") {
        addProcessAction("补充需求", "button--primary", () => focusDescription("请先说明要处理的主体；如果还想换背景或调色，也可以一起说。"));
        addProcessAction("改用手动选区", "button--secondary", switchToManualSelection);
      } else if (model.phase === "unsupported") {
        addProcessAction("修改需求", "button--secondary", () => focusDescription("可尝试换背景、主体调色、描边阴影或按主体裁切。"));
      } else if (model.phase === "failed") {
        addProcessAction("重新处理", "button--primary", startOneClick);
        addProcessAction("改用手动选区", "button--secondary", switchToManualSelection);
      } else if (model.phase === "completed") {
        addProcessDownload(editedDownloadLink, "下载 PNG");
        addProcessAction("继续微调", "button--secondary", openEditor);
        addProcessAction("开始新处理", "button--secondary", () => focusDescription("可修改描述后重新处理当前图片。"));
      }
      return;
    }

    if (model.source === "agent") {
      if (model.phase === "needs_choice") {
        addProcessAction("改用手动提示", "button--secondary", switchToManualSelection);
      } else if (model.phase === "needs_confirmation" || model.phase === "ready_to_segment") {
        addProcessAction("确认并生成选区", "button--primary", startAgentSegmentation);
        addProcessAction("改用手动提示", "button--secondary", switchToManualSelection);
      } else if (model.phase === "needs_manual_prompt") {
        addProcessAction("添加包含点", "button--primary", () => setMode("positive"));
        addProcessAction("框选目标", "button--secondary", () => setMode("box"));
      } else if (model.phase === "needs_refinement" || model.phase === "failed") {
        addProcessAction("添加包含点", "button--secondary", () => setMode("positive"));
        addProcessAction("框选细化", "button--secondary", () => setMode("box"));
        if (state.box || state.points.some((point) => point.label === 1)) {
          addProcessAction("使用新提示再试", "button--primary", startAgentSegmentation);
        }
      } else if (model.phase === "awaiting_evaluation") {
        addProcessAction("复核当前结果", "button--secondary", evaluateAgentResult);
      } else if (model.phase === "completed") {
        addProcessAction("继续微调", "button--secondary", openEditor);
      }
    }
  }

  function renderProcessHub() {
    const model = currentProcessPresentation();
    processHub.className = "process-hub process-hub--" + model.tone;
    processEyebrow.textContent = model.eyebrow;
    processTitle.textContent = model.title;
    processStage.textContent = model.stage;
    processMessage.textContent = model.message;
    processAssurance.textContent = model.assurance;
    processSteps.forEach((stepElement) => {
      const step = Number(stepElement.dataset.processStep);
      const complete = model.tone === "success" ? step <= model.step : step < model.step;
      stepElement.classList.toggle("is-complete", complete);
      stepElement.classList.toggle("is-current", !complete && step === model.step && model.step > 0);
    });
    renderProcessPlan(model.plan, model.planNotice);
    renderProcessActions(model);
  }

  function renderOneClickPanel() {
    renderProcessHub();
  }

  function renderAgentPanel() {
    renderProcessHub();
  }

  function applyOneClickCandidate(candidate) {
    const box = candidateToBox(candidate);
    if (!box) {
      return;
    }
    state.box = box;
    state.boxSource = "qwen";
    state.draftBox = null;
    state.boxDragStart = null;
    redraw();
  }

  function syncOneClickPlanToEditor(plan) {
    if (!plan || typeof plan !== "object") {
      return;
    }
    const selection = readPlanSection(plan, "selection");
    const background = readPlanSection(plan, "background");
    const subject = readPlanSection(plan, "subject");
    const effects = readPlanSection(plan, "effects");
    const crop = readPlanSection(plan, "crop");
    const assign = (input, value) => {
      if (!input || !Number.isFinite(Number(value))) return;
      input.value = String(Math.round(Number(value)));
    };
    assign(edgeOffset, selection.edge_offset);
    assign(featherPx, selection.feather_px);
    if (maskCleanup && typeof selection.cleanup === "boolean") maskCleanup.checked = selection.cleanup;
    if (backgroundMode && typeof background.mode === "string") backgroundMode.value = background.mode;
    if (backgroundColor && typeof background.color === "string" && /^#[0-9a-f]{6}$/i.test(background.color)) backgroundColor.value = background.color;
    assign(backgroundBlurPx, background.blur_px);
    assign(backgroundBrightness, background.brightness);
    assign(backgroundSaturation, background.saturation);
    if (backgroundGrayscale && typeof background.grayscale === "boolean") backgroundGrayscale.checked = background.grayscale;
    assign(subjectBrightness, subject.brightness);
    assign(subjectContrast, subject.contrast);
    assign(subjectSaturation, subject.saturation);
    assign(subjectHueDegrees, subject.hue_degrees);
    assign(subjectTemperature, subject.temperature);
    assign(subjectSharpen, subject.sharpen);
    assign(subjectBlurPx, subject.blur_px);
    assign(subjectOpacity, subject.opacity);
    assign(outlineWidthPx, effects.outline_width_px);
    if (outlineColor && typeof effects.outline_color === "string" && /^#[0-9a-f]{6}$/i.test(effects.outline_color)) outlineColor.value = effects.outline_color;
    assign(outlineOpacity, effects.outline_opacity);
    assign(shadowOffsetX, effects.shadow_offset_x);
    assign(shadowOffsetY, effects.shadow_offset_y);
    assign(shadowBlurPx, effects.shadow_blur_px);
    if (shadowColor && typeof effects.shadow_color === "string" && /^#[0-9a-f]{6}$/i.test(effects.shadow_color)) shadowColor.value = effects.shadow_color;
    assign(shadowOpacity, effects.shadow_opacity);
    if (cropEnabled && typeof crop.enabled === "boolean") cropEnabled.checked = crop.enabled;
    assign(cropPaddingPx, crop.padding_px);
    if (cropAspectRatio && typeof crop.aspect_ratio === "string") cropAspectRatio.value = crop.aspect_ratio;
    syncEditControls();
    expandPlanDrivenFolds(plan);
  }

  function applyOneClickRun(run) {
    if (!run || typeof run.run_id !== "string" || typeof run.phase !== "string") {
      throw new Error("一键处理没有返回有效状态，请重新试一次。");
    }
    state.oneClickRunId = run.run_id;
    state.oneClickPhase = run.phase;
    state.oneClickMessage = typeof run.message === "string" ? run.message : "正在处理。";
    state.oneClickPlan = run.plan && typeof run.plan === "object" ? run.plan : null;
    state.oneClickPlanNotice = typeof run.plan_notice === "string" ? run.plan_notice : "";
    state.oneClickCandidates = Array.isArray(run.candidates) ? run.candidates.slice(0, 3) : [];
    state.oneClickSelectedCandidateIndex = Number.isInteger(run.selected_candidate_index)
      ? run.selected_candidate_index
      : null;
    if (run.selected_candidate && typeof run.selected_candidate === "object") {
      applyOneClickCandidate(run.selected_candidate);
    }
    renderCandidateChoices();
    renderProcessHub();
  }

  function applyAgentRun(run) {
    if (!run || typeof run.agent_id !== "string" || typeof run.phase !== "string") {
      throw new Error("自动选区没有返回有效状态，请重新试一次。" );
    }
    state.agentRunId = run.agent_id;
    state.agentPhase = run.phase;
    state.agentMessage = typeof run.message === "string" ? run.message : "正在准备下一步。";
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
    renderCandidateChoices();
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
      setStatus("推荐位置无效，请重新定位或手动框选。", "error");
      return false;
    }
    const changed = !state.box || state.box.x0 !== box.x0 || state.box.y0 !== box.y0 || state.box.x1 !== box.x1 || state.box.y1 !== box.y1 || state.boxSource !== "qwen";
    state.box = box;
    state.boxSource = "qwen";
    state.draftBox = null;
    state.boxDragStart = null;
    state.selectedCandidateIndex = index;
    if (changed) {
      markSelectionDirty();
    }
    redraw();
    renderCandidateChoices();
    refreshActions();
    return true;
  }

  function renderCandidateChoices() {
    candidateChoices.replaceChildren();
    const oneClickNeedsChoice = state.oneClickRunId
      && state.oneClickPhase === "needs_target_confirmation";
    const agentNeedsChoice = state.agentRunId
      && ["needs_choice", "needs_confirmation"].includes(state.agentPhase);
    const candidates = oneClickNeedsChoice
      ? state.oneClickCandidates
      : agentNeedsChoice
        ? state.groundingCandidates
        : [];
    const selectedIndex = oneClickNeedsChoice
      ? state.oneClickSelectedCandidateIndex
      : state.selectedCandidateIndex;
    if (!candidates.length) {
      candidateChoices.hidden = true;
      return;
    }

    const heading = document.createElement("p");
    heading.className = "candidate-heading";
    heading.textContent = oneClickNeedsChoice
      ? candidates.length > 1
        ? "画面中有多个候选对象，请选一个再生成选区。"
        : "请确认这个候选对象，再生成选区。"
      : state.agentPhase === "needs_choice"
        ? "找到多个候选，请选一个。"
        : "找到一个候选，确认后生成选区。";
    candidateChoices.append(heading);

    const buttons = document.createElement("div");
    buttons.className = "candidate-buttons";
    candidates.forEach((candidate, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "candidate-button";
      const selected = index === selectedIndex;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
      button.disabled = state.busy;

      const label = typeof candidate.label === "string" && candidate.label.trim()
        ? candidate.label.trim()
        : "候选区域";
      button.textContent = "候选 " + String(index + 1) + " · " + label;
      button.addEventListener("click", async () => {
        if (oneClickNeedsChoice) {
          const box = candidateToBox(candidate);
          if (!box) {
            setStatus("推荐位置无效，请修改描述或改用手动选区。", "error");
            return;
          }
          state.oneClickSelectedCandidateIndex = index;
          applyOneClickCandidate(candidate);
          renderCandidateChoices();
          renderProcessHub();
          await chooseOneClickCandidate(index);
          return;
        }
        if (selectGroundingCandidate(index)) {
          setMode("box");
          if (state.agentRunId && state.agentPhase === "needs_choice") {
            await chooseAgentCandidate(index);
          } else {
            setStatus("已选候选 " + String(index + 1) + "，可生成或加点微调。", "success");
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
    const hadPrompts = state.points.length || state.box;
    if (!settings.skipHistory && hadPrompts) {
      rememberPromptSnapshot();
    }
    state.points = [];
    state.box = null;
    state.draftBox = null;
    state.boxDragStart = null;
    state.boxSource = null;
    clearGrounding();
    if (!settings.keepAgent) {
      clearAgentState();
    }
    if (settings.resetHistory) {
      state.promptUndo = [];
      state.promptRedo = [];
    } else if (hadPrompts) {
      markSelectionDirty();
    }
    redraw();
    refreshActions();
    if (!settings.quiet && state.imageId) {
      setStatus("提示已清空。请点选或框选主体。", "");
    }
  }

  function setMode(mode) {
    state.mode = mode;
    resetMaskToolButtons();
    modeButtons.forEach((button) => {
      const selected = button.dataset.mode === mode;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    if (mode === "positive") {
      modeHelp.textContent = "点主体内部，添加包含点。";
    } else if (mode === "negative") {
      modeHelp.textContent = "点要排除的位置。";
    } else {
      modeHelp.textContent = "拖框圈住目标，可配合点选。";
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

  function displayCanvasImage(image, sourceWidth, sourceHeight, options) {
    const settings = options || {};
    const width = Math.max(2, Math.round(sourceWidth || image.naturalWidth));
    const height = Math.max(2, Math.round(sourceHeight || image.naturalHeight));
    const scale = Math.min(1, maxCanvasEdge / Math.max(width, height));
    if (settings.asBase !== false) {
      state.baseImage = image;
      state.editImage = null;
      state.showingEdit = false;
    }
    state.image = image;
    if (!settings.preserveSourceDimensions) {
      state.width = width;
      state.height = height;
    }
    canvas.width = Math.max(2, Math.round(width * scale));
    canvas.height = Math.max(2, Math.round(height * scale));
    canvasEmpty.hidden = true;
    canvas.hidden = false;
    canvasShell.classList.add("canvas-shell--has-image");
    redraw();
    refreshActions();
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
            setStatus("正在上传图片…", "working");
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
    if (editorSection) {
      editorSection.hidden = true;
    }
    if (editResult) {
      editResult.hidden = true;
    }
    resetJobPoll();
    clearOneClickState();
    state.activeJobId = null;
    state.pollUrl = null;
    state.processJob = null;
    state.imageId = null;
    state.image = null;
    state.baseImage = null;
    state.editImage = null;
    state.width = 0;
    state.height = 0;
    state.resultId = null;
    state.selectionDirty = false;
    state.showingEdit = false;
    clearMaskOverlay();
    state.previewToken += 1;
    const previewToken = state.previewToken;
    safeSessionRemove(activeJobStorageKey);
    resetPrompts({ quiet: true, skipHistory: true, resetHistory: true });
    fileName.textContent = file.name;
    const form = new FormData();
    form.append("image", file);
    setBusy(true, "upload");
    setStatus("正在上传图片…", "working");
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
      setStatus("图片已上传。可一键处理或手动标注。", "success");
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
    setStatus("正在找推荐位置…", "working");

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
        setStatus(typeof result.note === "string" ? result.note : "没有找到候选。可手动点选或框选。", "error");
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
        ? "找到 " + String(candidates.length) + " 个候选，请选一个。" + note
        : "已推荐一个区域。可生成选区或加点微调。" + note, "success");
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
      setStatus("请先写下你想选取的主体。", "error");
      return;
    }

    clearGrounding();
    clearQwenBox();
    state.agentRunId = "pending";
    state.agentPhase = "locating";
    state.agentMessage = "正在查找主体。";
    state.agentEvaluation = null;
    redraw();
    renderAgentPanel();
    setBusy(true, "agent");
    setStatus("正在查找主体…", "working");

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

  function oneClickJobResult(run) {
    return run && run.job && typeof run.job === "object" ? run.job : null;
  }

  function showOneClickSegmentation(run) {
    const job = oneClickJobResult(run);
    if (!job || job.status !== "succeeded") {
      return false;
    }
    try {
      displayResult(job);
      return true;
    } catch (_error) {
      return false;
    }
  }

  async function finishOneClickRun(run) {
    const phase = run.phase;
    if (phase === "completed") {
      if (!showOneClickSegmentation(run)) {
        throw new Error("一键处理已完成，但没有找到可显示的选区结果。");
      }
      syncOneClickPlanToEditor(run.plan);
      await displayEditResult(run.edit);
      setBusy(false);
      setStatus("处理完成。可下载原图尺寸 PNG，或继续微调。", "success");
      if (editResult) editResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }
    if (phase === "selection_ready") {
      if (!showOneClickSegmentation(run)) {
        throw new Error("选区已生成，但没有找到可显示的预览。请刷新后重试。 ");
      }
      syncOneClickPlanToEditor(run.plan);
      setBusy(false);
      setStatus("选区已准备好。可直接检查边界，或继续微调。", "success");
      return;
    }
    if (phase === "needs_target_confirmation") {
      setBusy(false);
      setStatus(run.message || "请确认要处理的对象。", "");
      return;
    }
    if (phase === "needs_input") {
      showOneClickSegmentation(run);
      setBusy(false);
      setStatus(run.message || "请补充需求，或手动编辑。", "error");
      return;
    }
    if (phase === "unsupported") {
      setBusy(false);
      setStatus(run.message || "当前不支持这项处理。", "error");
      return;
    }
    if (phase === "failed") {
      showOneClickSegmentation(run);
      setBusy(false);
      setStatus(run.message || "处理未完成。可重试或手动编辑。", "error");
    }
  }

  async function enterOneClickReview(run) {
    if (!showOneClickSegmentation(run)) {
      throw new Error("选区已生成，但没有找到可显示的预览。请刷新后重试。 ");
    }
    syncOneClickPlanToEditor(run.plan);
    setBusy(false);
    setStatus(
      run.phase === "selection_ready"
        ? "选区已准备好。可直接检查边界，或继续微调。"
        : "选区和方案已准备好，请确认后生成预览。",
      "success",
    );
  }

  async function applyOneClickEdit() {
    if (!state.oneClickRunId || state.oneClickRunId === "pending") {
      return null;
    }
    const response = await fetch("/api/one-click-runs/" + encodeURIComponent(state.oneClickRunId) + "/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const run = await readResponse(response);
    applyOneClickRun(run);
    return run;
  }

  async function confirmOneClickPlan() {
    if (!state.oneClickRunId || state.oneClickRunId === "pending" || state.oneClickPhase !== "ready_to_apply") {
      return;
    }
    setBusy(true, "one-click-apply");
    state.oneClickPhase = "composing";
    state.oneClickMessage = "正在生成编辑预览。";
    renderProcessHub();
    try {
      const run = await applyOneClickEdit();
      if (!run) return;
      const job = oneClickJobResult(run);
      if (job) updateJobCard(job);
      if (run.phase === "segmenting" || run.phase === "composing") {
        await pollOneClickRun();
      } else {
        await finishOneClickRun(run);
      }
    } catch (error) {
      state.oneClickPhase = "failed";
      state.oneClickMessage = error.message || "生成预览失败。";
      setBusy(false);
      setStatus(state.oneClickMessage, "error");
    }
  }

  async function pollOneClickRun() {
    if (!state.oneClickRunId || state.oneClickRunId === "pending") {
      return;
    }
    try {
      const response = await fetch("/api/one-click-runs/" + encodeURIComponent(state.oneClickRunId), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      let run = await readResponse(response);
      state.oneClickPollFailures = 0;
      applyOneClickRun(run);
      const job = oneClickJobResult(run);
      if (job) updateJobCard(job);

      if (run.phase === "segmenting" || run.phase === "composing") {
        state.oneClickPollTimer = window.setTimeout(pollOneClickRun, pollIntervalMs);
        return;
      }
      if (run.phase === "selection_ready" || run.phase === "ready_to_apply") {
        await enterOneClickReview(run);
        return;
      }
      await finishOneClickRun(run);
    } catch (error) {
      state.oneClickPollFailures += 1;
      if (state.oneClickPollFailures >= 3) {
        setBusy(false);
        setStatus("暂时无法读取处理状态。请刷新页面确认结果。", "error");
        return;
      }
      state.oneClickPollTimer = window.setTimeout(pollOneClickRun, pollIntervalMs * state.oneClickPollFailures);
    }
  }

  async function startOneClick() {
    if (!state.imageId) {
      setStatus("请先选择一张图片。", "error");
      return;
    }
    if (!description.value.trim()) {
      setStatus("请先说明要处理的主体。", "error");
      return;
    }
    clearAgentState();
    clearGrounding();
    clearQwenBox();
    resetOneClickPoll();
    state.oneClickRunId = "pending";
    state.oneClickPhase = "planning";
    state.oneClickMessage = "正在准备处理。";
    state.oneClickPlan = null;
    state.oneClickPlanNotice = "";
    state.oneClickCandidates = [];
    state.oneClickSelectedCandidateIndex = null;
    state.processJob = null;
    redraw();
    renderOneClickPanel();
    setBusy(true, "one-click");
    setStatus("正在定位主体并生成选区…", "working");

    try {
      const response = await fetch("/api/one-click-runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: state.imageId, instruction: description.value.trim() }),
      });
      const run = await readResponse(response);
      applyOneClickRun(run);
      const job = oneClickJobResult(run);
      if (job) updateJobCard(job);
      if (run.phase === "segmenting" || run.phase === "composing") {
        await pollOneClickRun();
      } else if (run.phase === "selection_ready" || run.phase === "ready_to_apply") {
        await enterOneClickReview(run);
      } else {
        await finishOneClickRun(run);
      }
    } catch (error) {
      if (state.oneClickRunId === "pending") clearOneClickState();
      setBusy(false);
      setStatus(error.message, "error");
    }
  }

  async function chooseOneClickCandidate(index) {
    if (!state.oneClickRunId || state.oneClickRunId === "pending" || state.oneClickPhase !== "needs_target_confirmation") {
      return;
    }
    setBusy(true, "one-click");
    state.oneClickMessage = "已确认对象，正在生成选区。";
    renderProcessHub();
    try {
      const response = await fetch("/api/one-click-runs/" + encodeURIComponent(state.oneClickRunId) + "/choose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidate_index: index }),
      });
      const run = await readResponse(response);
      applyOneClickRun(run);
      const job = oneClickJobResult(run);
      if (job) updateJobCard(job);
      if (run.phase === "segmenting" || run.phase === "composing") {
        await pollOneClickRun();
      } else if (run.phase === "selection_ready" || run.phase === "ready_to_apply") {
        await enterOneClickReview(run);
      } else {
        await finishOneClickRun(run);
      }
    } catch (error) {
      state.oneClickMessage = error.message || "无法确认对象。";
      setBusy(false);
      setStatus(state.oneClickMessage, "error");
    }
  }

  async function chooseAgentCandidate(index) {
    if (!state.agentRunId || state.agentRunId === "pending") {
      return;
    }
    setBusy(true, "agent");
    setStatus("正在确认区域…", "working");
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
      message: job.message || fallbackMessage || "已提交，正在处理。",
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
    setStatus("正在生成选区…", "working");
    updateJobCard({ status: "queued", phase: "queued", message: "正在排队。" });
    try {
      const response = await fetch("/api/agent-runs/" + encodeURIComponent(state.agentRunId) + "/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(agentPromptPayload()),
      });
      const result = await readResponse(response);
      applyAgentRun(result);
      startTrackingJob(result.job, "已提交，正在处理。");
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
      setStatus("选区已生成；暂时无法完成自动质量复核。", "");
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
    setStatus("已提交，正在生成选区。", "working");
    updateJobCard({ status: "queued", phase: "queued", message: "正在排队。" });

    try {
      const response = await fetch("/api/segment-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSegmentationPayload()),
      });
      const result = await readResponse(response);
      startTrackingJob(result, "已提交，正在处理。");
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
      return { badge: "已完成", title: "选区已生成", progress: 100, state: "success" };
    }
    if (jobStatus === "failed") {
      return { badge: "未完成", title: "处理未完成", progress: 100, state: "error" };
    }
    if (jobStatus === "running") {
      const details = {
        loading_model: { title: "正在准备", progress: 30 },
        encoding_image: { title: "正在处理图片", progress: 48 },
        predicting: { title: "正在生成选区", progress: 68 },
        rendering: { title: "正在整理结果", progress: 86 },
      };
      const current = details[phase] || { title: "正在处理", progress: 56 };
      return { badge: "运行中", title: current.title, progress: current.progress, state: "running" };
    }
    return { badge: "排队中", title: "等待处理", progress: 18, state: "queued" };
  }

  function updateJobCard(job) {
    state.processJob = job && typeof job === "object" ? { ...job } : null;
    renderProcessHub();
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
    if (typeof result.result_id !== "string" || !result.result_id) {
      throw new Error("任务已经完成，但没有找到可编辑的选区编号。请重新生成选区。 ");
    }
    resultPreview.src = previewUrl;
    const area = num(result.mask_area_px);
    const iou = num(result.estimated_iou);
    const pieces = [];
    if (area !== null) {
      pieces.push("已选中约 " + Math.round(area).toLocaleString("zh-CN") + " 个像素");
    }
    if (iou !== null) {
      pieces.push("选区评分：" + iou.toFixed(3));
    }
    resultSummary.textContent = pieces.length ? pieces.join("；") + "。" : "选区已经生成。";
    maskLink.href = typeof result.mask_url === "string" ? result.mask_url : "#";
    contoursLink.href = typeof result.contours_url === "string" ? result.contours_url : "#";
    jsonLink.href = typeof result.json_url === "string" ? result.json_url : "#";
    [maskLink, contoursLink, jsonLink].forEach((link) => {
      link.classList.toggle("is-disabled", link.href.endsWith("/#"));
      link.setAttribute("aria-disabled", String(link.href.endsWith("/#")));
    });
    resultSection.hidden = false;
    state.resultId = result.result_id;
    state.selectionDirty = false;
    state.editImage = null;
    state.showingEdit = false;
    resetEditControls({ restoreCanvas: true });
    if (editorSection) {
      editorSection.hidden = false;
    }
    loadMaskOverlay(typeof result.mask_url === "string" ? result.mask_url : null, { resetStrokes: true });
    refreshActions();
  }

  function editNumber(input, fallback) {
    const value = input ? num(input.value) : null;
    return value === null ? fallback : Math.round(value);
  }

  function signedValue(value) {
    return value > 0 ? "+" + String(value) : String(value);
  }

  function numberSummary(label, value, unit) {
    return label + " " + signedValue(value) + (unit || "");
  }

  function pxSummary(label, value) {
    return label + " " + String(value) + " px";
  }

  function getFoldSummaryState() {
    const selectionSummary = [];
    const edge = editNumber(edgeOffset, 0);
    const feather = editNumber(featherPx, 0);
    if (state.maskStrokes.length) selectionSummary.push(String(state.maskStrokes.length) + " 条笔刷");
    if (!Boolean(maskCleanup && maskCleanup.checked)) selectionSummary.push("关闭清理");
    if (edge) selectionSummary.push(edge > 0 ? "扩展 " + String(edge) + " px" : "收缩 " + String(Math.abs(edge)) + " px");
    if (feather) selectionSummary.push("羽化 " + String(feather) + " px");

    const backgroundSummary = [];
    const mode = backgroundMode ? backgroundMode.value : "original";
    const backgroundModeCopy = { original: "原背景", transparent: "透明", color: "纯色", blur: "虚化" };
    backgroundSummary.push(backgroundModeCopy[mode] || "原背景");
    const backgroundBrightnessValue = editNumber(backgroundBrightness, 0);
    const backgroundSaturationValue = editNumber(backgroundSaturation, 0);
    if (mode === "color") backgroundSummary.push(validHexColor(backgroundColor && backgroundColor.value, "#FFFFFF"));
    if (mode === "blur") backgroundSummary.push(pxSummary("虚化", editNumber(backgroundBlurPx, 18)));
    if (backgroundBrightnessValue) backgroundSummary.push(numberSummary("亮度", backgroundBrightnessValue));
    if (backgroundSaturationValue) backgroundSummary.push(numberSummary("饱和度", backgroundSaturationValue));
    if (backgroundGrayscale && backgroundGrayscale.checked) backgroundSummary.push("灰度");

    const subjectSummary = [];
    const subjectBrightnessValue = editNumber(subjectBrightness, 0);
    const subjectContrastValue = editNumber(subjectContrast, 0);
    const subjectSaturationValue = editNumber(subjectSaturation, 0);
    const subjectHueValue = editNumber(subjectHueDegrees, 0);
    const subjectTemperatureValue = editNumber(subjectTemperature, 0);
    const subjectSharpenValue = editNumber(subjectSharpen, 0);
    const subjectBlurValue = editNumber(subjectBlurPx, 0);
    const subjectOpacityValue = editNumber(subjectOpacity, 100);
    if (subjectBrightnessValue) subjectSummary.push(numberSummary("亮度", subjectBrightnessValue));
    if (subjectContrastValue) subjectSummary.push(numberSummary("对比", subjectContrastValue));
    if (subjectSaturationValue) subjectSummary.push(numberSummary("饱和度", subjectSaturationValue));
    if (subjectHueValue) subjectSummary.push(numberSummary("色相", subjectHueValue, "°"));
    if (subjectTemperatureValue) subjectSummary.push(numberSummary("冷暖", subjectTemperatureValue));
    if (subjectSharpenValue) subjectSummary.push("锐化 " + String(subjectSharpenValue));
    if (subjectBlurValue) subjectSummary.push(pxSummary("模糊", subjectBlurValue));
    if (subjectOpacityValue !== 100) subjectSummary.push("透明度 " + String(subjectOpacityValue) + "%");

    const effectsSummary = [];
    const outlineWidthValue = editNumber(outlineWidthPx, 0);
    const outlineOpacityValue = editNumber(outlineOpacity, 100);
    const shadowXValue = editNumber(shadowOffsetX, 0);
    const shadowYValue = editNumber(shadowOffsetY, 0);
    const shadowBlurValue = editNumber(shadowBlurPx, 0);
    const shadowOpacityValue = editNumber(shadowOpacity, 0);
    if (outlineWidthValue) {
      effectsSummary.push(pxSummary("描边", outlineWidthValue));
      if (outlineOpacityValue !== 100) effectsSummary.push("描边透明度 " + String(outlineOpacityValue) + "%");
    }
    if (shadowXValue || shadowYValue || shadowBlurValue || shadowOpacityValue) {
      effectsSummary.push("阴影 " + String(shadowXValue) + "," + String(shadowYValue));
      if (shadowBlurValue) effectsSummary.push(pxSummary("模糊", shadowBlurValue));
      if (shadowOpacityValue) effectsSummary.push("阴影透明度 " + String(shadowOpacityValue) + "%");
    }

    const cropSummary = [];
    const cropEnabledValue = Boolean(cropEnabled && cropEnabled.checked);
    const cropPaddingValue = editNumber(cropPaddingPx, 24);
    const cropRatioValue = cropAspectRatio ? cropAspectRatio.value : "free";
    if (cropEnabledValue) {
      cropSummary.push("已裁切");
      cropSummary.push(pxSummary("留白", cropPaddingValue));
      cropSummary.push("比例 " + cropRatioValue);
    }

    return {
      selection: selectionSummary.length ? selectionSummary.join(" · ") : "保持原样",
      background: backgroundSummary.join(" · "),
      subject: subjectSummary.length ? subjectSummary.join(" · ") : "保持原样",
      effects: effectsSummary.length ? effectsSummary.join(" · ") : "无额外样式",
      crop: cropSummary.length ? cropSummary.join(" · ") : "不裁切",
    };
  }

  function syncEditControls() {
    const brush = editNumber(maskBrushRadius, 24);
    const edge = editNumber(edgeOffset, 0);
    const feather = editNumber(featherPx, 0);
    const backgroundBlur = editNumber(backgroundBlurPx, 18);
    const backgroundBrightnessLevel = editNumber(backgroundBrightness, 0);
    const backgroundSaturationLevel = editNumber(backgroundSaturation, 0);
    const brightness = editNumber(subjectBrightness, 0);
    const contrast = editNumber(subjectContrast, 0);
    const saturation = editNumber(subjectSaturation, 0);
    const hueDegrees = editNumber(subjectHueDegrees, 0);
    const temperature = editNumber(subjectTemperature, 0);
    const sharpen = editNumber(subjectSharpen, 0);
    const subjectBlur = editNumber(subjectBlurPx, 0);
    const opacity = editNumber(subjectOpacity, 100);
    const outlineWidth = editNumber(outlineWidthPx, 0);
    const outlineOpacityLevel = editNumber(outlineOpacity, 100);
    const shadowX = editNumber(shadowOffsetX, 0);
    const shadowY = editNumber(shadowOffsetY, 0);
    const shadowBlurLevel = editNumber(shadowBlurPx, 0);
    const shadowOpacityLevel = editNumber(shadowOpacity, 0);
    const cropPadding = editNumber(cropPaddingPx, 24);
    if (maskBrushRadiusValue) maskBrushRadiusValue.textContent = String(brush) + " px";
    if (edgeOffsetValue) edgeOffsetValue.textContent = edge === 0 ? "不偏移" : edge > 0 ? "扩展 +" + String(edge) + " px" : "收缩 " + String(Math.abs(edge)) + " px";
    if (featherPxValue) featherPxValue.textContent = String(feather) + " px";
    if (backgroundBlurPxValue) backgroundBlurPxValue.textContent = String(backgroundBlur) + " px";
    if (backgroundBrightnessValue) backgroundBrightnessValue.textContent = signedValue(backgroundBrightnessLevel);
    if (backgroundSaturationValue) backgroundSaturationValue.textContent = signedValue(backgroundSaturationLevel);
    if (subjectBrightnessValue) subjectBrightnessValue.textContent = signedValue(brightness);
    if (subjectContrastValue) subjectContrastValue.textContent = signedValue(contrast);
    if (subjectSaturationValue) subjectSaturationValue.textContent = signedValue(saturation);
    if (subjectHueDegreesValue) subjectHueDegreesValue.textContent = signedValue(hueDegrees) + "°";
    if (subjectTemperatureValue) subjectTemperatureValue.textContent = signedValue(temperature);
    if (subjectSharpenValue) subjectSharpenValue.textContent = String(sharpen);
    if (subjectBlurPxValue) subjectBlurPxValue.textContent = String(subjectBlur) + " px";
    if (subjectOpacityValue) subjectOpacityValue.textContent = String(opacity) + "%";
    if (outlineWidthPxValue) outlineWidthPxValue.textContent = String(outlineWidth) + " px";
    if (outlineOpacityValue) outlineOpacityValue.textContent = String(outlineOpacityLevel) + "%";
    if (shadowOffsetXValue) shadowOffsetXValue.textContent = signedValue(shadowX) + " px";
    if (shadowOffsetYValue) shadowOffsetYValue.textContent = signedValue(shadowY) + " px";
    if (shadowBlurPxValue) shadowBlurPxValue.textContent = String(shadowBlurLevel) + " px";
    if (shadowOpacityValue) shadowOpacityValue.textContent = String(shadowOpacityLevel) + "%";
    if (cropPaddingPxValue) cropPaddingPxValue.textContent = String(cropPadding) + " px";
    const mode = backgroundMode ? backgroundMode.value : "original";
    if (backgroundColorRow) backgroundColorRow.hidden = mode !== "color";
    if (backgroundBlurRow) backgroundBlurRow.hidden = mode !== "blur";
    const editorDisabled = state.busy || !editorAvailable();
    if (cropPaddingPx) cropPaddingPx.disabled = editorDisabled || !(cropEnabled && cropEnabled.checked);
    if (cropAspectRatio) cropAspectRatio.disabled = editorDisabled || !(cropEnabled && cropEnabled.checked);
    const summaries = getFoldSummaryState();
    if (foldSummarySelection) foldSummarySelection.textContent = summaries.selection;
    if (foldSummaryBackground) foldSummaryBackground.textContent = summaries.background;
    if (foldSummarySubject) foldSummarySubject.textContent = summaries.subject;
    if (foldSummaryEffects) foldSummaryEffects.textContent = summaries.effects;
    if (foldSummaryCrop) foldSummaryCrop.textContent = summaries.crop;
  }

  function resetEditControls(options) {
    const settings = options || {};
    if (maskBrushRadius) maskBrushRadius.value = "24";
    if (edgeOffset) edgeOffset.value = "0";
    if (featherPx) featherPx.value = "0";
    if (maskCleanup) maskCleanup.checked = true;
    if (backgroundMode) backgroundMode.value = "original";
    if (backgroundColor) backgroundColor.value = "#ffffff";
    if (backgroundBlurPx) backgroundBlurPx.value = "18";
    if (backgroundBrightness) backgroundBrightness.value = "0";
    if (backgroundSaturation) backgroundSaturation.value = "0";
    if (backgroundGrayscale) backgroundGrayscale.checked = false;
    if (subjectBrightness) subjectBrightness.value = "0";
    if (subjectContrast) subjectContrast.value = "0";
    if (subjectSaturation) subjectSaturation.value = "0";
    if (subjectHueDegrees) subjectHueDegrees.value = "0";
    if (subjectTemperature) subjectTemperature.value = "0";
    if (subjectSharpen) subjectSharpen.value = "0";
    if (subjectBlurPx) subjectBlurPx.value = "0";
    if (subjectOpacity) subjectOpacity.value = "100";
    if (outlineWidthPx) outlineWidthPx.value = "0";
    if (outlineColor) outlineColor.value = "#ffffff";
    if (outlineOpacity) outlineOpacity.value = "100";
    if (shadowOffsetX) shadowOffsetX.value = "0";
    if (shadowOffsetY) shadowOffsetY.value = "0";
    if (shadowBlurPx) shadowBlurPx.value = "0";
    if (shadowColor) shadowColor.value = "#000000";
    if (shadowOpacity) shadowOpacity.value = "0";
    if (cropEnabled) cropEnabled.checked = false;
    if (cropPaddingPx) cropPaddingPx.value = "24";
    if (cropAspectRatio) cropAspectRatio.value = "free";
    state.maskStrokes = [];
    state.activeMaskStroke = null;
    state.maskOverlayVisible = true;
    state.editImage = null;
    state.showingEdit = false;
    if (editResult) editResult.hidden = true;
    if (settings.restoreCanvas && state.baseImage) {
      displayCanvasImage(state.baseImage, state.width, state.height, { asBase: false });
    }
    if (state.mode === "mask-add" || state.mode === "mask-erase") {
      setMode("positive");
    } else {
      resetMaskToolButtons();
    }
    rebuildMaskOverlay();
    redraw();
    setFoldExpanded("selection", true, { skipAccordion: true });
    setFoldExpanded("background", !isSmallViewport(), { skipAccordion: true });
    setFoldExpanded("subject", false, { skipAccordion: true });
    setFoldExpanded("effects", false, { skipAccordion: true });
    setFoldExpanded("crop", false, { skipAccordion: true });
    syncEditControls();
    refreshActions();
  }

  function buildEditPayload() {
    return {
      image_id: state.imageId,
      result_id: state.resultId,
      selection: {
        strokes: state.maskStrokes.map((stroke) => ({
          mode: stroke.mode,
          radius: Math.round(Number(stroke.radius) || 24),
          points: stroke.points.map((point) => ({ x: point.x, y: point.y })),
        })),
        edge_offset: editNumber(edgeOffset, 0),
        feather_px: editNumber(featherPx, 0),
        cleanup: Boolean(maskCleanup && maskCleanup.checked),
      },
      background: {
        mode: backgroundMode ? backgroundMode.value : "original",
        color: backgroundColor ? backgroundColor.value : "#ffffff",
        blur_px: editNumber(backgroundBlurPx, 18),
        brightness: editNumber(backgroundBrightness, 0),
        saturation: editNumber(backgroundSaturation, 0),
        grayscale: Boolean(backgroundGrayscale && backgroundGrayscale.checked),
      },
      subject: {
        brightness: editNumber(subjectBrightness, 0),
        contrast: editNumber(subjectContrast, 0),
        saturation: editNumber(subjectSaturation, 0),
        hue_degrees: editNumber(subjectHueDegrees, 0),
        temperature: editNumber(subjectTemperature, 0),
        sharpen: editNumber(subjectSharpen, 0),
        blur_px: editNumber(subjectBlurPx, 0),
        opacity: editNumber(subjectOpacity, 100),
      },
      effects: {
        outline_width_px: editNumber(outlineWidthPx, 0),
        outline_color: outlineColor ? outlineColor.value : "#ffffff",
        outline_opacity: editNumber(outlineOpacity, 100),
        shadow_offset_x: editNumber(shadowOffsetX, 0),
        shadow_offset_y: editNumber(shadowOffsetY, 0),
        shadow_blur_px: editNumber(shadowBlurPx, 0),
        shadow_color: shadowColor ? shadowColor.value : "#000000",
        shadow_opacity: editNumber(shadowOpacity, 0),
      },
      crop: {
        enabled: Boolean(cropEnabled && cropEnabled.checked),
        padding_px: editNumber(cropPaddingPx, 24),
        aspect_ratio: cropAspectRatio ? cropAspectRatio.value : "free",
      },
    };
  }

  function editSummaryText(settings) {
    const pieces = [];
    const selectionSettings = settings && settings.selection && typeof settings.selection === "object" ? settings.selection : settings || {};
    const backgroundSettings = settings && settings.background && typeof settings.background === "object" ? settings.background : settings || {};
    const subjectSettings = settings && settings.subject && typeof settings.subject === "object" ? settings.subject : settings || {};
    const effectSettings = settings && settings.effects && typeof settings.effects === "object" ? settings.effects : settings || {};
    const cropSettings = settings && settings.crop && typeof settings.crop === "object" ? settings.crop : settings || {};
    const mode = backgroundSettings.background_mode || backgroundSettings.mode;
    const backgroundCopy = { original: "保留原背景", transparent: "透明背景", color: "纯色背景", blur: "背景虚化" };
    pieces.push(backgroundCopy[mode] || "局部编辑");
    if (settings && Array.isArray(settings.strokes) && settings.strokes.length) {
      pieces.push(String(settings.strokes.length) + " 条选区笔刷");
    }
    if (selectionSettings.edge_offset) {
      pieces.push(selectionSettings.edge_offset > 0 ? "边缘扩展" : "边缘收缩");
    }
    if (selectionSettings.feather_px) {
      pieces.push("羽化 " + String(selectionSettings.feather_px) + " px");
    }
    if (backgroundSettings.background_brightness || backgroundSettings.brightness) {
      pieces.push("背景亮度 " + signedValue(planInteger(backgroundSettings.background_brightness || backgroundSettings.brightness, 0)));
    }
    if (backgroundSettings.background_saturation || backgroundSettings.saturation) {
      pieces.push("背景饱和度 " + signedValue(planInteger(backgroundSettings.background_saturation || backgroundSettings.saturation, 0)));
    }
    if (backgroundSettings.background_grayscale || backgroundSettings.grayscale) {
      pieces.push("背景灰度");
    }
    if (subjectSettings.subject_brightness || subjectSettings.brightness) {
      pieces.push("亮度 " + signedValue(planInteger(subjectSettings.subject_brightness || subjectSettings.brightness, 0)));
    }
    if (subjectSettings.subject_contrast || subjectSettings.contrast) {
      pieces.push("对比度 " + signedValue(planInteger(subjectSettings.subject_contrast || subjectSettings.contrast, 0)));
    }
    if (subjectSettings.subject_saturation || subjectSettings.saturation) {
      pieces.push("饱和度 " + signedValue(planInteger(subjectSettings.subject_saturation || subjectSettings.saturation, 0)));
    }
    if (subjectSettings.subject_hue_degrees || subjectSettings.hue_degrees) {
      pieces.push("色相 " + signedValue(planInteger(subjectSettings.subject_hue_degrees || subjectSettings.hue_degrees, 0)) + "°");
    }
    if (subjectSettings.subject_temperature || subjectSettings.temperature) {
      pieces.push("冷暖 " + signedValue(planInteger(subjectSettings.subject_temperature || subjectSettings.temperature, 0)));
    }
    if (subjectSettings.subject_sharpen || subjectSettings.sharpen) {
      pieces.push("锐化 " + String(planInteger(subjectSettings.subject_sharpen || subjectSettings.sharpen, 0)));
    }
    if (subjectSettings.subject_blur_px || subjectSettings.blur_px) {
      pieces.push("柔化 " + String(planInteger(subjectSettings.subject_blur_px || subjectSettings.blur_px, 0)) + " px");
    }
    const opacityValue = subjectSettings.subject_opacity || subjectSettings.opacity;
    if (Number.isFinite(Number(opacityValue)) && planInteger(opacityValue, 100) !== 100) {
      pieces.push("透明度 " + String(planInteger(opacityValue, 100)) + "%");
    }
    if (effectSettings.outline_width_px) {
      pieces.push("描边 " + String(planInteger(effectSettings.outline_width_px, 0)) + " px");
    }
    if (effectSettings.shadow_offset_x || effectSettings.shadow_offset_y || effectSettings.shadow_blur_px || effectSettings.shadow_opacity) {
      pieces.push("已加阴影");
    }
    if (cropSettings.enabled || cropSettings.crop_enabled) {
      pieces.push("已裁切");
    }
    return pieces.join("；") + "。";
  }

  function displayEditOnCanvas(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        state.editImage = image;
        state.showingEdit = true;
        displayCanvasImage(image, image.naturalWidth, image.naturalHeight, {
          asBase: false,
          preserveSourceDimensions: true,
        });
        resolve();
      };
      image.onerror = () => reject(new Error("编辑图片已经生成，但浏览器未能显示预览。"));
      image.src = url;
    });
  }

  async function displayEditResult(result) {
    if (!result || typeof result.preview_url !== "string" || typeof result.download_url !== "string") {
      throw new Error("编辑任务没有返回完整结果，请重新试一次。 ");
    }
    editPreview.src = result.preview_url;
    editedDownloadLink.href = result.download_url;
    editedMaskLink.href = typeof result.mask_url === "string" ? result.mask_url : "#";
    [editedDownloadLink, editedMaskLink].forEach((link) => {
      link.classList.toggle("is-disabled", link.href.endsWith("/#"));
      link.setAttribute("aria-disabled", String(link.href.endsWith("/#")));
    });
    editSummary.textContent = editSummaryText(result.settings);
    editResult.hidden = false;
    await displayEditOnCanvas(result.preview_url);
    refreshActions();
  }

  async function applyEdit() {
    if (!editorAvailable()) {
      setStatus("请先完成或更新选区，再开始局部编辑。", "error");
      return;
    }
    const payload = buildEditPayload();
    if (payload.background.mode === "original" && payload.subject.opacity < 100) {
      setFoldExpanded("subject", true);
      setStatus("降低主体透明度时，请先选择透明、纯色或虚化背景。", "error");
      return;
    }
    setBusy(true, "edit");
    setStatus("正在合成高清编辑预览…", "working");
    try {
      const response = await fetch("/api/edits", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await readResponse(response);
      await displayEditResult(result);
      setBusy(false);
      setStatus(payload.crop.enabled
        ? "编辑预览已生成；下载按钮会导出裁切后的高清 PNG。"
        : "编辑预览已生成；下载按钮会导出原图尺寸 PNG。", "success");
    } catch (error) {
      setBusy(false);
      setStatus(error.message, "error");
    }
  }

  function toggleOriginalCanvas() {
    if (!state.editImage || !state.baseImage) {
      return;
    }
    if (state.showingEdit) {
      state.showingEdit = false;
      displayCanvasImage(state.baseImage, state.width, state.height, { asBase: false });
    } else {
      state.showingEdit = true;
      displayCanvasImage(state.editImage, state.editImage.naturalWidth, state.editImage.naturalHeight, {
        asBase: false,
        preserveSourceDimensions: true,
      });
    }
    refreshActions();
  }

  function finishJobWithError(message) {
    resetJobPoll();
    state.activeJobId = null;
    state.pollUrl = null;
    state.jobStartedAt = null;
    safeSessionRemove(activeJobStorageKey);
    setBusy(false);
    updateJobCard({ status: "failed", message: message || "处理未完成。" });
    setStatus(message || "处理未完成。请调整提示后重试。", "error");
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
      setStatus("选区已生成。可下载 Mask、轮廓或 JSON。", "success");
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
        finishJobWithError("暂时无法读取任务状态。请刷新页面确认。 ");
        return;
      }
      updateJobCard({ status: "running", message: "正在重连…" });
      state.pollTimer = window.setTimeout(pollJob, pollIntervalMs * state.pollFailures);
    }
  }

  async function loadGroundingStatus() {
    try {
      const response = await fetch("/api/grounding/status", { cache: "no-store" });
      const result = await readResponse(response);
      state.groundingAvailable = Boolean(result.configured || result.ready);
      if (state.groundingAvailable) {
        groundingStatus.textContent = "自动定位可用。";
        groundingStatus.className = "engine-note engine-note--ready";
      } else {
        groundingStatus.textContent = "自动定位未配置；仍可点选或框选。";
        groundingStatus.className = "engine-note";
      }
    } catch (_error) {
      state.groundingAvailable = false;
      groundingStatus.textContent = "暂无法确认自动定位；仍可手动选区。";
      groundingStatus.className = "engine-note";
    }
    refreshActions();
  }

  async function loadRuntimeStatus() {
    try {
      const response = await fetch("/api/runtime/status", { cache: "no-store" });
      const result = await readResponse(response);
      const ready = Boolean(result.ready);
      runtimeChip.className = "runtime-chip" + (ready ? " runtime-chip--ready" : " runtime-chip--warning");
      runtimeChip.lastElementChild.textContent = ready ? "服务正常" : "服务暂不可用";
    } catch (_error) {
      runtimeChip.className = "runtime-chip runtime-chip--warning";
      runtimeChip.lastElementChild.textContent = "服务暂不可用";
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
      updateJobCard({ status: "running", message: "正在恢复任务…" });
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
    if (state.oneClickRunId && state.oneClickRunId !== "pending") {
      clearOneClickState();
    }
    if (state.groundingId || state.agentRunId) {
      const clearsQwenBox = state.boxSource === "qwen";
      clearGrounding();
      clearAgentState();
      if (clearsQwenBox) {
        clearQwenBox();
        redraw();
      }
      setStatus("描述已改，请重新分析图片。", "");
    }
    refreshActions();
  });

  modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  resetPromptsButton.addEventListener("click", () => resetPrompts());
  if (undoPromptButton) undoPromptButton.addEventListener("click", undoPrompt);
  if (redoPromptButton) redoPromptButton.addEventListener("click", redoPrompt);
  groundButton.addEventListener("click", autoGround);
  runButton.addEventListener("click", startAgent);
  if (oneClickButton) oneClickButton.addEventListener("click", startOneClick);
  segmentButton.addEventListener("click", startSegmentation);
  if (maskAddButton) maskAddButton.addEventListener("click", () => setMaskTool("mask-add"));
  if (maskEraseButton) maskEraseButton.addEventListener("click", () => setMaskTool("mask-erase"));
  if (undoMaskStrokeButton) undoMaskStrokeButton.addEventListener("click", undoMaskStroke);
  if (clearMaskStrokesButton) clearMaskStrokesButton.addEventListener("click", clearMaskStrokes);
  if (toggleMaskOverlayButton) {
    toggleMaskOverlayButton.addEventListener("click", () => {
      state.maskOverlayVisible = !state.maskOverlayVisible;
      redraw();
      refreshActions();
    });
  }
  if (showOriginalButton) showOriginalButton.addEventListener("click", toggleOriginalCanvas);
  if (applyEditButton) applyEditButton.addEventListener("click", applyEdit);
  if (resetEditButton) {
    resetEditButton.addEventListener("click", () => {
      if (!state.resultId || state.busy) return;
      resetEditControls({ restoreCanvas: true });
      setStatus("编辑设置已恢复默认；原始选区仍然保留。", "");
    });
  }
  editorFoldToggles.forEach((toggle) => {
    toggle.addEventListener("click", () => {
      const root = toggle.closest("[data-editor-fold]");
      const name = root ? root.getAttribute("data-editor-fold") : null;
      if (!name) {
        return;
      }
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      setFoldExpanded(name, !expanded);
    });
  });
  editorInputs.forEach((input) => {
    input.addEventListener("input", syncEditControls);
    input.addEventListener("change", syncEditControls);
  });
  window.addEventListener("resize", () => {
    if (!isSmallViewport()) {
      return;
    }
    const expandedNames = Object.keys(editorInputGroups).filter((name) => {
      const elements = foldElements(name);
      return elements && elements.toggle.getAttribute("aria-expanded") === "true";
    });
    expandedNames.slice(1).forEach((name) => setFoldExpanded(name, false, { skipAccordion: true }));
  });

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.image || !state.imageId || state.busy) {
      return;
    }
    if (state.showingEdit) {
      setStatus("当前显示编辑预览；先点“查看原图”再调整选区。", "");
      return;
    }
    const point = pointFromEvent(event);
    if (state.mode === "mask-add" || state.mode === "mask-erase") {
      beginMaskStroke(point, event.pointerId);
    } else if (state.mode === "box") {
      state.boxDragStart = point;
      state.draftBox = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
      canvas.setPointerCapture(event.pointerId);
    } else {
      rememberPromptSnapshot();
      if (state.agentRunId && state.agentPhase === "needs_choice") {
        clearAgentState();
        clearGrounding();
        setStatus("已切换到手动提示。包含点会直接用于生成选区。", "");
      }
      state.points.push({ x: point.x, y: point.y, label: state.mode === "positive" ? 1 : 0 });
      markSelectionDirty();
      refreshActions();
    }
    redraw();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (state.activeMaskStroke && !state.busy && state.imageId) {
      extendMaskStroke(pointFromEvent(event));
      return;
    }
    if (!state.draftBox || !state.boxDragStart || state.busy || !state.imageId) {
      return;
    }
    state.draftBox = normalizedBox(state.boxDragStart, pointFromEvent(event));
    scheduleRedraw();
  });

  canvas.addEventListener("pointerup", (event) => {
    if (state.activeMaskStroke && !state.busy && state.imageId) {
      extendMaskStroke(pointFromEvent(event));
      completeMaskStroke(event.pointerId);
      return;
    }
    if (!state.draftBox || !state.boxDragStart || state.busy || !state.imageId) {
      return;
    }
    const completed = normalizedBox(state.boxDragStart, pointFromEvent(event));
    state.draftBox = null;
    state.boxDragStart = null;
    if (completed.x1 - completed.x0 >= 2 && completed.y1 - completed.y0 >= 2) {
      rememberPromptSnapshot();
      state.box = completed;
      state.boxSource = "manual";
      const keepsManualAgent = state.agentRunId && state.agentPhase === "needs_manual_prompt";
      if (!keepsManualAgent) {
        clearAgentState();
      }
      clearGrounding();
      markSelectionDirty();
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
    if (state.activeMaskStroke) {
      state.activeMaskStroke = null;
      rebuildMaskOverlay();
    }
    state.draftBox = null;
    state.boxDragStart = null;
    redraw();
  });

  window.addEventListener("keydown", (event) => {
    const target = event.target;
    const isTyping = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement || (target instanceof HTMLElement && target.isContentEditable);
    if (isTyping) {
      return;
    }
    if (event.key === "Escape" && (state.draftBox || state.activeMaskStroke)) {
      state.draftBox = null;
      state.boxDragStart = null;
      state.activeMaskStroke = null;
      rebuildMaskOverlay();
      redraw();
      return;
    }
    if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "z") {
      return;
    }
    event.preventDefault();
    if (event.shiftKey) {
      redoPrompt();
    } else if ((state.mode === "mask-add" || state.mode === "mask-erase") && state.maskStrokes.length) {
      undoMaskStroke();
    } else {
      undoPrompt();
    }
  });

  window.addEventListener("beforeunload", () => {
    resetJobPoll();
    resetOneClickPoll();
    clearLocalPreview();
  });

  setFoldExpanded("selection", true, { skipAccordion: true });
  setFoldExpanded("background", !isSmallViewport(), { skipAccordion: true });
  setFoldExpanded("subject", false, { skipAccordion: true });
  setFoldExpanded("effects", false, { skipAccordion: true });
  setFoldExpanded("crop", false, { skipAccordion: true });
  setMode("positive");
  syncEditControls();
  renderOneClickPanel();
  updateDescriptionCount();
  loadGroundingStatus();
  loadRuntimeStatus();
  restoreActiveJob();
})();
