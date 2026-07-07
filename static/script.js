function convertirLinks(texto) {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    return texto.replace(urlRegex, function(url) {
        return `<a href="${url}" target="_blank">${url}</a>`;
    });
}

const input = document.getElementById("input");
const chat = document.getElementById("chat");

// enviar con Enter
input.addEventListener("keypress", function(e) {
    if (e.key === "Enter") {
        enviar();
    }
});

async function enviar() {
    const mensaje = input.value;
    if (mensaje.trim() === "") return;

    // ELIMINAR BOTONES ANTERIORES
    document.querySelectorAll(".opciones-container").forEach(container => container.remove());

    chat.innerHTML += `<div class="mensaje-user">${mensaje}</div>`;
    input.value = "";

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            mensaje: mensaje
        })
    });

    const data = await response.json();
    const confianza = data.confianza ?? 0;

    chat.innerHTML += `
        <div class="mensaje-bot">
            ${convertirLinks(data.respuesta)}<br>
            <small>Fuente: ${data.fuente}</small>
            <small>Confianza: ${(confianza * 100).toFixed(1)}%</small>
        </div>
    `;

    // --- RENDERIZADO DINÁMICO DE OPCIONES LIMPIO (Sin estilos en línea) ---
    if (data.opciones && data.opciones.length > 0) {
        let botonesHtml = `<div class="opciones-container">`;
        
        data.opciones.forEach(opcion => {
            botonesHtml += `
                <button class="btn-opcion" onclick="enviarOpcion('${opcion}')">
                    ${opcion}
                </button>`;
        });
        
        botonesHtml += `</div>`;
        chat.innerHTML += botonesHtml;
    }

    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: "smooth"
    });
}

// --- NUEVA FUNCIÓN GLOBAL PARA LOS CLICS DE LOS BOTONES ---
window.enviarOpcion = function(opcion) {
    input.value = opcion; // Ponemos el texto de la opción en el input
    enviar();            // Reutilizamos la función nativa de enviar
};