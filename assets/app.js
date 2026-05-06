// ============================================================
//  app.js — language toggle, original-zh cache, PDF link swap
// ============================================================
(function () {
  "use strict";

  var STORAGE_KEY = "site-lang";
  var btn = document.getElementById("lang-toggle");
  var cvLink = document.getElementById("cv-download");
  var html = document.documentElement;

  // 1. 把 HTML 里写的中文当作 zh 字典缓存起来,避免维护两份
  var zhCache = {};
  document.querySelectorAll("[data-i18n]").forEach(function (el) {
    var key = el.getAttribute("data-i18n");
    zhCache[key] = el.innerHTML;
  });

  function applyLang(lang) {
    var dict = lang === "en" ? (window.I18N && window.I18N.en) || {} : zhCache;

    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      var key = el.getAttribute("data-i18n");
      if (Object.prototype.hasOwnProperty.call(dict, key)) {
        el.innerHTML = dict[key];
      }
    });

    html.setAttribute("lang", lang === "en" ? "en" : "zh-CN");

    // 切换按钮显示文字(显示"切到另一种语言"的标签)
    document.querySelectorAll("[data-lang-label]").forEach(function (el) {
      var match = el.getAttribute("data-lang-label");
      // 当前 zh -> 显示 "EN" 按钮; 当前 en -> 显示 "中文"
      el.hidden = !((lang === "zh" && match === "zh") || (lang === "en" && match === "en"));
    });

    // PDF 链接跟随语言
    if (cvLink) {
      cvLink.setAttribute(
        "href",
        lang === "en" ? "./assets/resume_en.pdf" : "./assets/resume_zh.pdf"
      );
    }

    try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}
  }

  function currentLang() {
    var saved;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
    if (saved === "en" || saved === "zh") return saved;
    // 浏览器偏好回退
    var nav = (navigator.language || "").toLowerCase();
    return nav.indexOf("zh") === 0 ? "zh" : "en";
  }

  // 2. 初始应用
  var initLang = currentLang();
  if (initLang !== "zh") applyLang(initLang);
  else applyLang("zh"); // 同步按钮显示

  // 3. 绑定切换
  if (btn) {
    btn.addEventListener("click", function () {
      var next = (html.getAttribute("lang") || "").toLowerCase().indexOf("zh") === 0 ? "en" : "zh";
      applyLang(next);
    });
  }

  // 4. 微信二维码:存在 ./assets/wechat.png 时才显示
  var wechatCard = document.getElementById("wechat-card");
  var wechatImg = document.getElementById("wechat-qr");
  if (wechatCard && wechatImg) {
    var probe = new Image();
    probe.onload = function () { wechatCard.hidden = false; };
    probe.onerror = function () { /* 文件不存在则保持 hidden */ };
    probe.src = wechatImg.src;
  }
})();
