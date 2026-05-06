(function () {
  "use strict";

  var STORAGE_KEY = "site-lang";
  var btn = document.getElementById("lang-toggle");
  var cvLink = document.getElementById("cv-download");
  var html = document.documentElement;

  // 把 HTML 内嵌的中文当作 zh 字典缓存
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

    document.querySelectorAll("[data-lang-label]").forEach(function (el) {
      var match = el.getAttribute("data-lang-label");
      el.hidden = !((lang === "zh" && match === "zh") || (lang === "en" && match === "en"));
    });

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
    var nav = (navigator.language || "").toLowerCase();
    return nav.indexOf("zh") === 0 ? "zh" : "en";
  }

  applyLang(currentLang());

  if (btn) {
    btn.addEventListener("click", function () {
      var next = (html.getAttribute("lang") || "").toLowerCase().indexOf("zh") === 0 ? "en" : "zh";
      applyLang(next);
    });
  }
})();