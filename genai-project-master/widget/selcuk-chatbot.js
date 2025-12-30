(function(){
  // ✅ Backend API adresin
  const API_URL = "http://localhost:8787/chat"; // PROD: https://senin-domainin/chat
  const API_ORIGIN = new URL(API_URL).origin;   // ✅ http://localhost:8787

  const launcher = document.createElement("button");
  launcher.id = "selcuk-chatbot-launcher";
  launcher.setAttribute("aria-label","Chatbot");
  launcher.innerHTML = `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3C6.5 3 2 6.8 2 11.5c0 2.6 1.4 5 3.7 6.7L5 21l3.4-1.6c1.1.3 2.3.6 3.6.6 5.5 0 10-3.8 10-8.5S17.5 3 12 3zm-4 9h8v2H8v-2zm0-3h8v2H8V9z"></path>
    </svg>
  `;

  const panel = document.createElement("div");
  panel.id = "selcuk-chatbot-panel";
  panel.innerHTML = `
    <div id="selcuk-chatbot-header">
      <div class="title">Selçuk Asistan <span class="badge">Beta</span></div>
      <button id="selcuk-chatbot-close" aria-label="Kapat">×</button>
    </div>
    <div id="selcuk-chatbot-messages"></div>
    <div id="selcuk-chatbot-inputbar">
      <input id="selcuk-chatbot-input" placeholder="Sorunu yaz..." />
      <button id="selcuk-chatbot-send">Gönder</button>
    </div>
    <div id="selcuk-chatbot-footerhint">
      Yalnızca Selçuk Üniversitesi yönetmelik / işlemleri hakkında yanıt verir.
    </div>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const closeBtn = panel.querySelector("#selcuk-chatbot-close");
  const msgBox = panel.querySelector("#selcuk-chatbot-messages");
  const input = panel.querySelector("#selcuk-chatbot-input");
  const sendBtn = panel.querySelector("#selcuk-chatbot-send");

  let open = false;
  let history = [];

  // -----------------------
  // HELPERS
  // -----------------------
  function normalizeSources(sources){
    // backend: sources: [{name, url}, ...]
    if (!sources) return [];
    if (Array.isArray(sources)) return sources;
    return [];
  }

  function absolutizeUrl(url){
    if (!url) return "";
    // ✅ /docs/... gibi relative gelirse absolute yap
    if (url.startsWith("/")) return API_ORIGIN + url;
    return url;
  }

  function renderSourcesBox(sources){
    const src = normalizeSources(sources);
    if (!src.length) return null;

    const box = document.createElement("div");
    box.className = "sources-box";

    const title = document.createElement("div");
    title.className = "sources-title";
    title.textContent = "Kaynak:";
    box.appendChild(title);

    const list = document.createElement("div");
    list.className = "sources-list";

    src.forEach((item, idx) => {
      // item bazen string de gelebilir diye toleranslıyız
      const name = (item && typeof item === "object") ? (item.name || "") : String(item || "");
      let url  = (item && typeof item === "object") ? (item.url || "")  : "";

      url = absolutizeUrl(url);

      if (url) {
        const a = document.createElement("a");
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener";
        a.className = "source-link";
        a.textContent = name || url;
        list.appendChild(a);
      } else {
        const span = document.createElement("span");
        span.className = "source-text";
        span.textContent = name;
        list.appendChild(span);
      }

      if (idx < src.length - 1) {
        const sep = document.createElement("span");
        sep.className = "source-sep";
        sep.textContent = ", ";
        list.appendChild(sep);
      }
    });

    box.appendChild(list);
    return box;
  }

  // -----------------------
  // UI RENDER
  // -----------------------
  function addMessage(role, text, sources){
    const wrap = document.createElement("div");
    wrap.className = "msg " + (role === "user" ? "user" : "assistant");

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    // ✅ kaynak mini-kutusu bubble içine
    const srcBox = renderSourcesBox(sources);
    if (srcBox) bubble.appendChild(srcBox);

    wrap.appendChild(bubble);
    msgBox.appendChild(wrap);
    msgBox.scrollTop = msgBox.scrollHeight;

    return wrap;
  }

  function updateMessage(wrapEl, text, sources){
    if (!wrapEl) return;

    const bubble = wrapEl.querySelector(".bubble");
    if (!bubble) return;

    // bubble içeriğini sıfırla (text + sources)
    bubble.textContent = text;

    const oldBox = bubble.querySelector(".sources-box");
    if (oldBox) oldBox.remove();

    const srcBox = renderSourcesBox(sources);
    if (srcBox) bubble.appendChild(srcBox);

    msgBox.scrollTop = msgBox.scrollHeight;
  }

  // -----------------------
  // SEND MESSAGE
  // -----------------------
  async function send(){
    const text = (input.value || "").trim();
    if(!text) return;

    addMessage("user", text);
    history.push({ role:"user", content:text });
    input.value = "";

    const typingEl = addMessage("assistant", "Yazıyorum...");

    try{
      const res = await fetch(API_URL, {
        method:"POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ message: text, history })
      });

      if (!res.ok) {
        updateMessage(typingEl, "Bir hata oluştu. (Sunucu yanıt vermedi)");
        return;
      }

      const data = await res.json();
      const answer = data.answer || "Bir hata oluştu.";
      const sources = data.sources || [];

      updateMessage(typingEl, answer, sources);
      history.push({ role:"assistant", content: answer });

    } catch(e){
      updateMessage(typingEl, "Bağlantı hatası. Daha sonra tekrar dene.");
    }
  }

  // -----------------------
  // EVENTS
  // -----------------------
  launcher.addEventListener("click", () => {
    open = !open;
    panel.style.display = open ? "block" : "none";

    if (open && msgBox.childElementCount === 0){
      addMessage(
        "assistant",
        "Merhaba 👋 Selçuk Üniversitesi ile ilgili bir sorunuz varsa yardımcı olabilirim."
      );
    }
    if (open) input.focus();
  });

  closeBtn.addEventListener("click", () => {
    open = false;
    panel.style.display = "none";
  });

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") send();
  });

})();
