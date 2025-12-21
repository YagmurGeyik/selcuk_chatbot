(function(){
  // ✅ Backend API adresin
  const API_URL = "http://localhost:8787/chat"; // PROD: https://senin-domainin/chat

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
    <div id="selcuk-chatbot-footerhint">Yalnızca Selçuk Üniversitesi yönetmelik / işlemleri hakkında yanıt verir.</div>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const closeBtn = panel.querySelector("#selcuk-chatbot-close");
  const msgBox = panel.querySelector("#selcuk-chatbot-messages");
  const input = panel.querySelector("#selcuk-chatbot-input");
  const sendBtn = panel.querySelector("#selcuk-chatbot-send");

  let open = false;
  let history = [];

  function addMessage(role, text){
    const wrap = document.createElement("div");
    wrap.className = "msg " + (role === "user" ? "user" : "assistant");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    msgBox.appendChild(wrap);
    msgBox.scrollTop = msgBox.scrollHeight;
  }

  async function send(){
    const text = (input.value || "").trim();
    if(!text) return;

    addMessage("user", text);
    history.push({role:"user", content:text});
    input.value = "";

    addMessage("assistant", "Yazıyorum...");
    const typingEl = msgBox.lastChild;

    try{
      const res = await fetch(API_URL, {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ message: text, history })
      });
      const data = await res.json();
      typingEl.querySelector(".bubble").textContent = data.answer || "Bir hata oluştu.";
      history.push({role:"assistant", content: data.answer || ""});
    }catch(e){
      typingEl.querySelector(".bubble").textContent = "Bağlantı hatası. Daha sonra tekrar dene.";
    }
    msgBox.scrollTop = msgBox.scrollHeight;
  }

  launcher.addEventListener("click", () => {
    open = !open;
    panel.style.display = open ? "block" : "none";
    if(open && msgBox.childElementCount === 0){
      addMessage("assistant", "Merhaba 👋 Selçuk Üniversitesi ile ilgili bir sorunuz varsa yardımcı olabilirim.");
    }
    if(open) input.focus();
  });

  closeBtn.addEventListener("click", () => {
    open = false;
    panel.style.display = "none";
  });

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if(e.key === "Enter") send();
  });
})();
