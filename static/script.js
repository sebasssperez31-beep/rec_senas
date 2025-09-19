document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("startRec");
  const stopBtn = document.getElementById("stopRec");
  const resultDiv = document.getElementById("resultado");

  // ================================
  // PLAY / STOP (Reconocimiento)
  // ================================
  if (startBtn && stopBtn) {
    startBtn.addEventListener("click", () => {
      fetch("/start_recognition", { method: "POST" })
        .then(() => {
          console.log("Reconocimiento iniciado");
          resultDiv.innerText = "▶ Reconocimiento iniciado...";
        });
    });

    stopBtn.addEventListener("click", () => {
      fetch("/stop_recognition", { method: "POST" })
        .then(() => {
          console.log("Reconocimiento detenido");
          resultDiv.innerText = "⏹ Reconocimiento detenido.";
        });
    });
  }

  // ================================
  // ACTUALIZAR RESULTADO
  // ================================
  const historialList = document.getElementById("historial");
  let vocalEntrenando = null;

  setInterval(() => {
    fetch("/status")
      .then(res => res.json())
      .then(data => {
        if (data.label) {
          resultDiv.innerText = `👉 Señal detectada: ${data.label}`;

          // Si estoy entrenando una vocal y coincide con la detectada
          if (vocalEntrenando && data.label === vocalEntrenando) {
            stopTimer(); // detener cronómetro
            console.log(`✔ Vocal ${vocalEntrenando} lograda en ${tiempo}s`);

            // ✅ Resetear color a verde y añadir check
            if (tiempoSpan) {
              tiempoSpan.style.color = "green";
              tiempoSpan.textContent = `${tiempo}s ✔`;
            }

            // Agregar al historial
            if (historialList) {
              const li = document.createElement("li");
              li.textContent = `Vocal ${vocalEntrenando} → ${tiempo}s`;
              historialList.appendChild(li);
            }

            vocalEntrenando = null; // limpiar estado
          }
        }
      })
      .catch(err => console.error("Error en /status:", err));
  }, 1000);

  // ================================
  // ENTRENAMIENTO (antiguo con input)
  // ================================
  const startTrainBtn = document.getElementById("startTrain");
  const stopTrainBtn = document.getElementById("stopTrain");
  const labelInput = document.getElementById("labelInput");

  if (startTrainBtn && stopTrainBtn && labelInput) {
    startTrainBtn.addEventListener("click", () => {
      const label = labelInput.value.trim();
      if (!label) {
        alert("Por favor escribe un nombre para la seña");
        return;
      }
      const formData = new FormData();
      formData.append("label", label);

      fetch("/start_training", {
        method: "POST",
        body: formData
      }).then(() => {
        console.log("Entrenamiento iniciado con etiqueta:", label);
        resultDiv.innerText = `📸 Entrenando con: ${label}...`;
      });
    });

    stopTrainBtn.addEventListener("click", () => {
      fetch("/stop_training", { method: "POST" })
        .then(() => {
          console.log("Entrenamiento detenido");
          resultDiv.innerText = "✅ Entrenamiento detenido.";
        });
    });
  }

  // ================================
  // NUEVO: Cronómetro e Intentos
  // ================================
  let tiempo = 0;
  let intentos = 0;
  let timerInterval = null;

  const tiempoSpan = document.getElementById("tiempo");
  const intentosSpan = document.getElementById("intentos");
  const cameraSection = document.getElementById("camera-section");

  // Ocultar cámara al inicio
  if (cameraSection) {
    cameraSection.style.display = "none";
  }

  // Función para iniciar cronómetro (con colores)
  function startTimer() {
    tiempo = 0;
    if (tiempoSpan) {
      tiempoSpan.textContent = "0s";
      tiempoSpan.style.color = "green"; // arranca en verde
    }
    timerInterval = setInterval(() => {
      tiempo++;
      if (tiempoSpan) {
        tiempoSpan.textContent = `${tiempo}s`;
        // Cambiar color según tiempo
        if (tiempo < 10) {
          tiempoSpan.style.color = "green";
        } else if (tiempo < 20) {
          tiempoSpan.style.color = "orange";
        } else {
          tiempoSpan.style.color = "red";
        }
      }
    }, 1000);
  }

  // Función para detener cronómetro (corregida ✅)
  function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  // ================================
  // NUEVO: Entrenar vocales (botones dinámicos)
  // ================================
  const trainButtons = document.querySelectorAll(".btn-train");

  trainButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const vocal = btn.getAttribute("data-vocal");
      vocalEntrenando = vocal; // guardar vocal en curso

      // Mostrar cámara
      if (cameraSection) {
        cameraSection.style.display = "flex";
      }
      // Incrementar intentos
      intentos++;
      if (intentosSpan) intentosSpan.textContent = intentos;
      // Iniciar cronómetro
      startTimer();
      console.log(`🔄 Entrenando vocal: ${vocal}`);
    });
  });

  // Cuando se detiene el entrenamiento (global stop)
  if (stopTrainBtn) {
    stopTrainBtn.addEventListener("click", () => {
      stopTimer();
      if (cameraSection) {
        cameraSection.style.display = "none";
      }
      vocalEntrenando = null; // cancelar entrenamiento manualmente
    });
  }
});
