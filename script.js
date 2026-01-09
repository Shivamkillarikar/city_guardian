let latitude = null, longitude = null;
let map, marker, lightTiles, darkTiles;

// ---------------- IMAGE PREVIEW ----------------
document.getElementById("image").addEventListener("change", (e) => {
    const file = e.target.files[0];
    const preview = document.getElementById("imagePreview");
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.classList.remove("d-none");
    }
});

// ---------------- LOCATION + MAP ----------------
navigator.geolocation.getCurrentPosition(async (pos) => {
    latitude = pos.coords.latitude;
    longitude = pos.coords.longitude;

    const address = await reverseGeocode(latitude, longitude);
    document.getElementById("locationInfo").innerText = address;
    document.getElementById("locationSkeleton").classList.add("d-none");
    document.getElementById("mapSkeleton").classList.add("d-none");
    document.getElementById("locationInfo").classList.remove("d-none");
    document.getElementById("map").style.display = "block";

    setTimeout(() => {
        map = L.map("map").setView([latitude, longitude], 15);
        lightTiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OSM" });
        darkTiles = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { attribution: "© OSM" });

        const isDark = document.body.classList.contains("dark-mode");
        (isDark ? darkTiles : lightTiles).addTo(map);

        marker = L.marker([latitude, longitude]).addTo(map).bindPopup("Current Location").openPopup();
        map.invalidateSize();
    }, 200);
}, () => {
    document.getElementById("locationInfo").innerText = "Location permission denied";
});

async function reverseGeocode(lat, lng) {
    const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`);
    const data = await res.json();
    formData.append("address",data.display_name)
    return data.display_name;
}

// ---------------- HISTORY & PDF LOGIC ----------------
function saveToHistory(data) {
    const history = JSON.parse(localStorage.getItem("cg_history") || "[]");
    history.unshift({ id: Date.now(), date: new Date().toLocaleString(), ...data });
    localStorage.setItem("cg_history", JSON.stringify(history.slice(0, 5)));
    renderHistory();
}

function renderHistory() {
    const container = document.getElementById("historyList");
    const history = JSON.parse(localStorage.getItem("cg_history") || "[]");
    if (history.length === 0) return;
    container.innerHTML = history.map(item => `
        <div class="list-group-item px-0 border-0 border-bottom">
            <div class="d-flex justify-content-between align-items-center mb-1">
                <small class="text-primary fw-bold">${item.department}</small>
                <span class="badge ${item.urgency === 'high' ? 'bg-danger' : 'bg-warning'} text-dark text-capitalize">${item.urgency}</span>
            </div>
            <p class="mb-0 small text-truncate">${item.complaint}</p>
            <small class="text-muted" style="font-size: 10px;">${item.date}</small>
        </div>
    `).join('');
}

function generatePDF(name, complaint, dept) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.setFont("helvetica", "bold");
    doc.text("CITYGUARDIAN REPORT RECEIPT", 20, 20);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.text(`Date: ${new Date().toLocaleString()}`, 20, 30);
    doc.line(20, 35, 190, 35);
    doc.text(`Reporter: ${name}`, 20, 50);
    doc.text(`Department: ${dept}`, 20, 60);
    doc.text("Description:", 20, 75);
    doc.text(doc.splitTextToSize(complaint, 160), 20, 85);
    doc.save(`Receipt_${Date.now()}.pdf`);
}

// ---------------- SUBMIT ----------------
document.getElementById("reportForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!latitude) { showToast("Location required", "warning"); return; }

    const formData = new FormData();
    formData.append("name", document.getElementById("name").value);
    formData.append("email", document.getElementById("email").value);
    formData.append("complaint", document.getElementById("complaint").value);
    formData.append("latitude", latitude);
    formData.append("longitude", longitude);
    if (document.getElementById("image").files[0]) {
        formData.append("image", document.getElementById("image").files[0]);
    }

    showSpinner(true);
    try {
        const res = await fetch("https://city-guardian.onrender.com/send-report", { method: "POST", body: formData });
        const data = await res.json();

        if (res.ok && data.status === "success") {
            showToast("Success! Downloading Receipt...", "success");
            saveToHistory({ complaint: complaint.value, department: data.department, urgency: data.urgency });
            generatePDF(name.value, complaint.value, data.department);
            document.getElementById("reportForm").reset();
            document.getElementById("imagePreview").classList.add("d-none");
        } else {
            showToast(data.message || "Submission failed", "danger");
        }
    } catch (err) {
        showToast("Server unreachable", "danger");
    } finally {
        showSpinner(false);
    }
});

// ---------------- UI HELPERS ----------------
function showSpinner(show) {
    document.getElementById("spinner").classList.toggle("d-none", !show);
    document.getElementById("submitText").innerText = show ? "Processing..." : "Submit Report";
    document.getElementById("submitBtn").disabled = show;
}

function showToast(message, type) {
    const container = document.querySelector(".toast-container");
    const toast = document.createElement("div");
    toast.className = `toast align-items-center text-bg-${type} border-0 show mb-2`;
    toast.innerHTML = `<div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

const themeToggle = document.getElementById("themeToggle");
themeToggle.onclick = () => {
    document.body.classList.toggle("dark-mode");
    const isDark = document.body.classList.contains("dark-mode");
    localStorage.setItem("theme", isDark ? "dark" : "light");
    themeToggle.innerText = isDark ? "☀️ Light" : "🌙 Dark";
    if (map) {
        map.eachLayer(l => map.removeLayer(l));
        (isDark ? darkTiles : lightTiles).addTo(map);
        marker.addTo(map);
    }
};

window.onload = () => {
    if(localStorage.getItem("theme") === "dark") {
        document.body.classList.add("dark-mode");
        themeToggle.innerText = "☀️ Light";
    }
    renderHistory();
};


// ---------------- SPEECH TO TEXT ----------------
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.continuous = false; // Stops automatically when you finish speaking
    recognition.interimResults = false;

    const voiceBtn = document.getElementById("voiceBtn");

    voiceBtn.onclick = () => {
        try {
            recognition.start();
        } catch (err) {
            console.log("Recognition already started or blocked");
        }
    };

    recognition.onstart = () => {
        voiceBtn.classList.replace("btn-outline-secondary", "btn-danger");
        voiceBtn.innerText = "🛑 Recording... Speak now";
        showToast("Listening... Please speak clearly", "info");
    };

    recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        document.getElementById("complaint").value = transcript;
        showToast("Voice captured successfully!", "success");
    };

    recognition.onerror = (e) => {
        console.error("Speech error:", e.error);
        if (e.error === 'not-allowed') {
            showToast("Microphone access blocked. Enable it in browser settings.", "danger");
        } else {
            showToast("Speech recognition failed. Try again.", "warning");
        }
        resetVoiceBtn();
    };

    recognition.onend = () => {
        resetVoiceBtn();
    };

    function resetVoiceBtn() {
        voiceBtn.classList.replace("btn-danger", "btn-outline-secondary");
        voiceBtn.innerText = "🎤 Voice Input";
    }
} else {
    document.getElementById("voiceBtn").disabled = true;
    document.getElementById("voiceBtn").innerText = "🎤 Voice not supported";

}


