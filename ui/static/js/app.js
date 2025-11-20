document.addEventListener("DOMContentLoaded", () => {
  const imageForm = document.getElementById("image-form");
  const videoForm = document.getElementById("video-form");
  const imageStatus = document.getElementById("image-status");
  const videoStatus = document.getElementById("video-status");
  const facesGrid = document.getElementById("faces-grid");
  const imageVerdict = document.getElementById("image-verdict");
  const videoVerdict = document.getElementById("video-verdict");
  const imageStats = document.getElementById("image-stats");
  const videoStats = document.getElementById("video-stats");
  const timelineCanvas = document.getElementById("timeline-chart");

  let timelineChart = null;

  const setStatus = (element, message, isError = false) => {
    if (!element) return;
    element.textContent = message;
    element.style.color = isError ? "#a41623" : "#2f4858";
  };

  const updateImageSummary = (summary) => {
    if (!summary) return;
    imageVerdict.textContent = summary.final_verdict.replace("_", " ");
    const stats = [
      `Faces inspected: ${summary.face_count ?? 0}`,
      `Fake faces: ${summary.fake_faces ?? 0}`,
      `Real faces: ${summary.real_faces ?? 0}`,
      `Aggregate score: ${
        summary.aggregate_score !== null ? summary.aggregate_score : "—"
      }`,
    ];
    imageStats.innerHTML = stats.map((item) => `<li>${item}</li>`).join("");
  };

  const renderFaces = (faces = []) => {
    if (!faces.length) {
      facesGrid.innerHTML =
        "<p>No faces catalogued yet. Upload an image to begin.</p>";
      return;
    }

    facesGrid.innerHTML = faces
      .map(
        (face, idx) => `
        <div class="face-card">
            <h4>Face ${idx + 1}: ${face.verdict}</h4>
            <p><strong>Final Score:</strong> ${face.final_score}</p>
            <p><strong>CNN Fake Probability:</strong> ${face.prob_fake}</p>
            <p><strong>AE Reconstruction Error:</strong> ${face.ae_loss}</p>
            <p><strong>Face Detection Confidence:</strong> ${face.confidence}</p>
        </div>
      `
      )
      .join("");
  };

  const updateTimeline = (timeline = []) => {
    if (!timeline || !timelineCanvas) return;

    const labels = timeline.map((entry) => `${entry.timestamp}s`);
    const data = timeline.map((entry) =>
      entry.score !== null ? entry.score : null
    );

    const chartData = {
      labels,
      datasets: [
        {
          label: "Aggregate Fake Score",
          data,
          borderColor: "#222222",
          backgroundColor: "rgba(34, 34, 34, 0.05)",
          spanGaps: true,
          tension: 0.3,
          fill: true,
        },
      ],
    };

    const config = {
      type: "line",
      data: chartData,
      options: {
        responsive: true,
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: { stepSize: 0.1 },
          },
        },
        plugins: {
          legend: { display: false },
        },
      },
    };

    if (timelineChart) {
      timelineChart.destroy();
    }
    timelineChart = new Chart(timelineCanvas, config);
  };

  const updateVideoSummary = (payload) => {
    if (!payload) return;
    videoVerdict.textContent = payload.final_verdict;
    const stats = [
      `Frames analyzed: ${payload.frames_analyzed ?? 0}`,
      `Final score: ${
        payload.final_score !== null ? payload.final_score : "—"
      }`,
      `Timeline entries: ${payload.timeline?.length ?? 0}`,
    ];
    videoStats.innerHTML = stats.map((item) => `<li>${item}</li>`).join("");
  };

  imageForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(imageStatus, "Uploading image…");
    const formData = new FormData(imageForm);

    try {
      const response = await fetch("/detect-image", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok || payload.status !== "success") {
        throw new Error(payload.message || "Image detection failed.");
      }
      setStatus(imageStatus, "Analysis complete.");
      updateImageSummary(payload.summary);
      renderFaces(payload.faces);
    } catch (error) {
      console.error(error);
      setStatus(imageStatus, error.message, true);
      renderFaces([]);
    }
  });

  videoForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(videoStatus, "Uploading footage…");
    const formData = new FormData(videoForm);

    try {
      const response = await fetch("/detect-video", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok || payload.status !== "success") {
        throw new Error(payload.message || "Video detection failed.");
      }
      setStatus(videoStatus, "Timeline ready.");
      updateVideoSummary(payload);
      updateTimeline(payload.timeline);
    } catch (error) {
      console.error(error);
      setStatus(videoStatus, error.message, true);
    }
  });
});

