document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("startBtn");
  const trainBtn = document.getElementById("trainBtn");
  const stopBtn = document.getElementById("stopBtn");
  const video = document.getElementById("videoFeed");
  const labelInput = document.getElementById("labelInput");
  const currentLabel = document.getElementById("currentLabel");
  const history = document.getElementById("history");

  let cameraOn = false;

  // 👉 función auxiliar para historial
  function addToHistory(message) {
    const li = document.createElement("li");
    li.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
    history.prepend(li); // prepend para que lo más nuevo quede arriba
  }

  // 👉 Solo mostrar la cámara con puntos
  startBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/start_recognition", { method: "POST" });
      if (!res.ok) throw new Error("Error al iniciar cámara");

      video.src = "/video";
      video.style.display = "block";
      cameraOn = true;

      addToHistory("Cámara iniciada");
    } catch (err) {
      alert("Error al conectar con el servidor.");
      console.error(err);
    }
  });

  // 👉 Comienza a guardar datos con la etiqueta escrita
  trainBtn.addEventListener("click", async () => {
    const label = labelInput.value.trim();
    if (!label) {
      alert("Escribe una etiqueta antes de entrenar.");
      return;
    }
    try {
      const res = await fetch("/start_training", {
        method: "POST",
        body: new URLSearchParams({ label }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      if (!res.ok) throw new Error("Error al iniciar entrenamiento");

      currentLabel.textContent = label;
      labelInput.disabled = true;
      addToHistory(`Entrenando con etiqueta: ${label}`);
    } catch (err) {
      alert("Error al conectar con el servidor.");
      console.error(err);
    }
  });

  // 👉 Detener cámara y entrenamiento
  stopBtn.addEventListener("click", async () => {
    try {
      if (cameraOn) {
        await fetch("/stop_recognition", { method: "POST" });
        video.style.display = "none";
        video.src = "";
        cameraOn = false;
        addToHistory("Cámara detenida");
      }

      await fetch("/stop_training", { method: "POST" });
      labelInput.disabled = false;
      currentLabel.textContent = "";
      addToHistory("Entrenamiento detenido");
    } catch (err) {
      alert("Error al conectar con el servidor.");
      console.error(err);
    }
  });

  // Entrenar todas las señas
document.addEventListener("DOMContentLoaded", function () {
  const trainAllBtn = document.getElementById("trainAllBtn");

  if (trainAllBtn) {
    trainAllBtn.addEventListener("click", function () {
      // Redirige a la ruta /train_all en Flask
      window.location.href = "/train_all";
    });
  }
});

});
