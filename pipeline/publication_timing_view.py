"""Dashboard-only presentation for sourced publication timing histograms."""

PANEL = """
<section id="panel-publications" role="tabpanel" hidden>
<article class="card">
  <div class="card-head"><div><h2>Wann veröffentlichen die S&amp;P-500-Unternehmen?</h2>
  <p>Kalendertage relativ zum offiziellen Quartalsende (31.03., 30.06., 30.09., 31.12.).
  Erste Ergebnisveröffentlichung und SEC-Berichtseinreichung sind unterschiedliche Ereignisse.</p></div></div>
  <div class="filters"><label for="publication-scope">Zeitraum</label>
    <select id="publication-scope">
      <option value="quarter">Ausgewähltes Quartal</option>
      <option value="history">Historisches Muster vor diesem Quartal</option>
    </select>
  </div>
  <p id="publication-context" class="small"></p>
  <p class="table-note">Unbekannt bedeutet nicht „noch nicht veröffentlicht“.
  Aktuelle Quartale sind unvollständig beobachtet. Grüne Modellmarkierungen spielen hier keine Rolle:
  gezählt wird das gesamte belegte Quartalsuniversum, nicht nur die ML-Auswahl.</p>
</article>
<div id="publication-charts" class="publication-grid"></div>
<article class="card"><h2>Datenabdeckung und Grenzen</h2>
  <div id="publication-coverage"></div>
  <p class="small">Unternehmen werden über ihre CIK nur einmal pro Quartal gezählt;
  Aktienklassen werden nicht als zusätzliche Unternehmen gezählt. Nicht aufgelöste Zuordnungen
  bleiben separat unbestätigt. Historische Balken zählen Unternehmensquartale, nicht verschiedene Unternehmen.</p>
  <p class="small">Market Cap ist die belegte USD-Marktkapitalisierung des gesamten Emittenten
  über alle Aktienklassen zum jeweiligen offiziellen Quartalsende. Keine SPY-Gewichte, kein
  heutiger Ersatzwert und keine pauschale Hochrechnung fehlender Unternehmen.
  Historisch werden USD-Unternehmensquartale aufsummiert, keine heutige Indexgewichtung.</p>
  <details><summary>Methodik und Quellenhinweise</summary><ul id="publication-notes"></ul></details>
</article>
</section>
"""

STYLE = """
.publication-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.publication-grid .card{min-width:0}
.publication-scroll{overflow-x:auto}
.publication-svg{width:100%;min-width:560px;display:block}
.publication-grid h3{font-size:15px;margin:0 0 8px}
.publication-empty{padding:28px 12px;background:#fff6e7;border-radius:8px;color:#754610}
.publication-grid .small{line-height:1.6}
@media(max-width:1000px){.publication-grid{grid-template-columns:1fr}}
"""

SCRIPT = r"""
function publicationChart(series, cap, eventLabel, buckets){
  const metric=cap?'Marktkapitalisierung':'Unternehmenszahl';
  const values=cap?series.market_cap_usd:series.counts;
  const count=cap?series.joint_observations:series.known_dates;
  const unit=cap?'Mrd. USD':series.historical?'Unternehmensquartale':'Unternehmen';
  const format=v=>cap?new Intl.NumberFormat('de-DE',{maximumFractionDigits:3}).format(v/1e9):num(v);
  const coverage=`${num(count)} / ${num(series.expected)} ${series.historical?'Unternehmensquartale':'erwartete Emittentenfälle'} mit ${cap?'belegtem Termin und Market Cap':'belegtem Termin'}`;
  if(!count)return `<article class="card"><h3>${E(eventLabel)} · ${metric}</h3><p class="small">${E(coverage)}</p><div class="publication-empty">${cap?'Keine belegten Termin-/Market-Cap-Paare vorhanden. Fehlende Werte sind nicht null; die Abdeckungstabelle trennt fehlende Termine und Market Caps.':'Keine belegten Termine für dieses Ereignis im lokalen Datensatz. Andere Ereignisse oder geschätzte Termine werden dafür nicht eingesetzt.'}</div></article>`;
  const width=720,height=290,left=56,right=12,top=30,bottom=62;
  const chartWidth=width-left-right,chartHeight=height-top-bottom;
  const maximum=Math.max(...values),step=chartWidth/buckets.length;
  const bars=values.map((value,i)=>{
    const h=maximum?value/maximum*chartHeight:0,x=left+i*step+4,y=top+chartHeight-h;
    const tip=`${buckets[i].label} Tage: ${format(value)} ${unit}`;
    return `<g><rect x="${x}" y="${y}" width="${step-8}" height="${h}" fill="${cap?'#608bc2':'#087f83'}"><title>${E(tip)}</title></rect><text x="${x+(step-8)/2}" y="${y-5}" text-anchor="middle" font-size="9" fill="#172c3a">${value?format(value):''}</text><text x="${x+(step-8)/2}" y="${height-bottom+18}" text-anchor="middle" font-size="9" fill="#61717d">${E(buckets[i].label)}</text></g>`;
  }).join('');
  const ticks=[0,.25,.5,.75,1].map(f=>{const y=top+chartHeight*(1-f);return `<line x1="${left}" x2="${width-right}" y1="${y}" y2="${y}" stroke="#e2e8ec"/><text x="${left-6}" y="${y+3}" text-anchor="end" font-size="10" fill="#61717d">${format(maximum*f)}</text>`;}).join('');
  return `<article class="card"><h3>${E(eventLabel)} · ${metric}</h3><p class="small">${E(coverage)}</p><div class="publication-scroll"><svg class="publication-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${E(eventLabel+' nach '+metric)}"><title>${E(eventLabel+' · '+metric)}</title>${ticks}${bars}<text x="${left}" y="14" font-size="10" fill="#61717d">${E(unit)}</text><text x="${width/2}" y="${height-12}" text-anchor="middle" font-size="11" fill="#61717d">Kalendertage nach offiziellem Quartalsende</text></svg></div>${cap?`<p class="small">${series.full_cap_coverage?'Market Caps für das gesamte erfasste Universum vorhanden.':'Nur bekannte Market Caps; keine vollständige S&P-500-Kapitalisierungsverteilung.'} Summe mit belegtem Termin: ${format(series.dated_cap_usd)} Mrd. USD.</p>`:''}</article>`;
}
function renderPublications(){
  const report=data.publication_timing;
  const context=document.getElementById('publication-context');
  const charts=document.getElementById('publication-charts');
  const coverage=document.getElementById('publication-coverage');
  const notes=document.getElementById('publication-notes');
  if(!report?.available){
    context.textContent=report?.error||'Keine Publikationsdaten für diese Umgebung.';
    charts.innerHTML='';coverage.textContent='Keine Auswertung verfügbar.';notes.innerHTML='';return;
  }
  const historical=document.getElementById('publication-scope').value==='history';
  const group=historical?report.history:report.quarter;
  const cycles=historical?report.history.cycles:[report.cycle_key];
  context.textContent=`${historical?'Historisches Muster':'Quartalsansicht'}: ${cycles.join(', ')||'keine früheren Quartale verfügbar'} · Lokale Auswertung ${dt(report.generated_at)} (kein Liveabruf). ${historical?'Jeweilige Quartalsmitglieder; lokale CIK-Zuordnung historisch nicht vollständig zeitversioniert.':`Offizielles Quartalsende ${date(report.quarter.quarter_end)}; Beobachtung bis Tag ${report.quarter.observation_days_after_end}.`}`;
  charts.innerHTML=[['sec','SEC-Berichtseinreichung (10-Q / 10-K)'],['earnings','Erste Earnings-Veröffentlichung']].map(([key,label])=>publicationChart(group[key],false,label,report.buckets)+publicationChart(group[key],true,label,report.buckets)).join('');
  coverage.innerHTML='<div class="table-wrap"><table><thead><tr><th>Ereignis</th><th>Termin bekannt</th><th>Termin unbekannt</th><th>Market Cap bekannt</th><th>Termin + Cap</th></tr></thead><tbody>'+[['sec','SEC-Bericht'],['earnings','Erste Earnings-Veröffentlichung']].map(([key,label])=>{const s=group[key];return `<tr><td>${label}</td><td>${num(s.known_dates)} / ${num(s.expected)}</td><td>${num(s.missing_dates)}</td><td>${num(s.known_caps)} / ${num(s.expected)}</td><td>${num(s.joint_observations)}</td></tr>`;}).join('')+'</tbody></table></div>';
  notes.innerHTML=report.notes.map(note=>`<li class="small">${E(note)}</li>`).join('');
}
document.getElementById('publication-scope').addEventListener('change',renderPublications);
if(location.hash==='#publications')go('publications');
"""


def extend_html(html: str) -> str:
    replacements = [
        ('<section id="panel-reports"', PANEL + '<section id="panel-reports"'),
        ('<button role="tab" aria-selected="false" data-tab="reports"',
         '<button role="tab" data-tab="publications" aria-selected="false">Veröffentlichungen</button>'
         '<button role="tab" aria-selected="false" data-tab="reports"'),
        ("if(state.tab==='reports')renderReports();",
         "if(state.tab==='publications')renderPublications();if(state.tab==='reports')renderReports();"),
        ("function go(tab){",
         "function go(tab){if(tab==='publications')history.replaceState(null,'','#publications');"
         "else if(location.hash==='#publications')"
         "history.replaceState(null,'',location.pathname+location.search);"),
        ("</style>", STYLE + "</style>"),
        ("</script>", SCRIPT + "</script>"),
    ]
    for old, new in replacements:
        if html.count(old) != 1:
            raise RuntimeError(f"Publication chart template anchor missing or ambiguous: {old}")
        html = html.replace(old, new, 1)
    return html
