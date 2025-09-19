let intervalId = null;

function actualizarProgreso() {
  fetch("/status_practica")
    .then(res => res.json())
    .then(data => {
      document.getElementById("intentos").innerText = data.intentos;
      document.getElementById("correctos").innerText = data.correctos;
      document.getElementById("racha").innerText = data.racha;
      document.getElementById("precision").innerText = data.precision.toFixed(2) + "%";
    });
}

document.getElementById("playBtn").addEventListener("click", () => {
  if (!intervalId) {
    intervalId = setInterval(actualizarProgreso, 1000);
  }
});

document.getElementById("stopBtn").addEventListener("click", () => {
  clearInterval(intervalId);
  intervalId = null;
});

document.getElementById("resetBtn").addEventListener("click", () => {
  fetch("/reiniciar_practica", {method: "POST"})
    .then(() => actualizarProgreso());
});
