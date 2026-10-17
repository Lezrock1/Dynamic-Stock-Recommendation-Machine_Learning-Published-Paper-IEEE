"""Dependency-free interactive research total-return chart."""

PANEL = """
<section id="panel-performance" role="tabpanel" hidden>
<article class="card"><h2>Historische Fortsetzung &amp; Total Return</h2>
<p>USD vor persönlicher Anlegerbesteuerung. Nur Reihen mit geprüfter Quelldefinition;
Paper-Referenz, Forschungsportfolio und Live-Depot werden nicht zusammengefügt.</p>
<div id="performance-errors" class="small"></div>
<div class="filters">
<label for="instrument-search">Instrumentensuche (Name, Ticker, ISIN, WKN, Indexkennung)</label>
<input id="instrument-search" type="search" placeholder="Eindeutige Kennung oder Name">
</div><div id="performance-instruments" class="performance-instruments"></div>
<p class="table-note">Mehrere Treffer bleiben getrennt auswählbar. Keine WKN-zu-Ticker-Heuristik.
Ohne berechtigten Anbieter verwendet die Suche ausschließlich den geprüften lokalen Instrumentenstamm.
Eine unbekannte Kennung löst keinen stillen Ersatz aus.</p>
<div class="filters">
<label for="performance-from">Von</label><input id="performance-from" type="date">
<label for="performance-to">Bis</label><input id="performance-to" type="date">
<label for="performance-metric">Darstellung</label>
<select id="performance-metric"><option value="index100">Startwert 100</option>
<option value="return">Kumulierte Rendite</option><option value="drawdown">Drawdown</option></select>
</div>
<div id="performance-presets" class="performance-presets">
<button data-range="1M">1M</button><button data-range="3M">3M</button>
<button data-range="6M">6M</button><button data-range="YTD">YTD</button>
<button data-range="1J">1J</button><button data-range="3J">3J</button>
<button data-range="5J">5J</button><button data-range="Alles">Alles</button>
<button id="performance-quarter">Gewähltes Quartal / Halteperiode</button>
<button id="performance-export">Sichtbare Daten als CSV</button></div>
<p id="performance-range" class="small"></p>
<div class="performance-chart" id="performance-chart">
<svg id="performance-svg" viewBox="0 0 1000 360" role="img" aria-label="Total-Return-Vergleich"></svg>
</div><div id="performance-hover" class="performance-hover"></div>
<p class="tiny muted">Mausrad: Zoom um den Cursor · Ziehen: Zeitraum verschieben.
Ein gemeinsamer Cursor zeigt die Werte aller sichtbaren Reihen am selben Bewertungszeitpunkt.</p>
<div class="table-wrap"><table id="performance-metrics"></table></div>
<details><summary>Vergleichsdefinitionen und Quellen der sichtbaren Reihen</summary>
<div id="performance-definitions"></div></details>
</article>
<article class="card"><h2>Historische Gewichte am ausgewählten Datum</h2>
<label for="historical-weight-date">Bewertungstag</label><input type="date" id="historical-weight-date">
<div id="historical-weight-note" class="small"></div>
<div class="table-wrap"><table id="historical-weights"></table></div>
<p class="table-note">Nur zum Datum belegte Modell-Zielgewichte und simulierte Bestände.
Keine heutigen Benchmarkgewichte zurückrechnen, keine Renormierung auf sichtbare Aktien.
Live-Bestände bleiben in der Depotansicht. Ohne Live-Ausführung: noch nicht investiert.</p></article>
<article class="card"><h2>Paper-Regeln, Anschluss und Quartalsstatus</h2>
<h3>Zielportfolio-Pakete · keine Orders</h3><div id="target-package-list"></div>
<div id="historical-source-progress" class="small"></div>
<details><summary>Neue Q2-Rechendiagnose · keine Handelsanweisung</summary>
<div id="numeric-q2-diagnostic"></div></details>
<div id="fixed-lag-summary"></div>
<div class="table-wrap"><table id="fixed-lag-quarters"></table></div>
<details><summary>Paper → bisheriger Code → Fortsetzung → Abweichung</summary>
<div class="table-wrap"><table id="fixed-lag-rules"></table></div></details>
</article></section>
"""

STYLE = """
.performance-presets{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}
.performance-instruments{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:9px}
.performance-instruments label{font-size:12px;border:1px solid var(--line);padding:10px;border-radius:8px}
.performance-instruments small{display:block;color:var(--sub);margin-top:5px}
.performance-chart{min-height:160px;touch-action:none;user-select:none}
#performance-svg{width:100%;min-height:230px;display:block}
.performance-hover{font-size:12px;min-height:28px;display:flex;gap:14px;flex-wrap:wrap}
#performance-errors{color:#9b5b13}
#fixed-lag-quarters td{white-space:normal;min-width:95px}
#fixed-lag-rules td{white-space:normal;min-width:180px}
"""

SCRIPT = r"""
const performanceState={selected:new Set(),initialized:false,view:null,drag:null,holdingsDate:null};
const performanceColors=['#087f83','#608bc2','#bb7433','#8057ac','#cc5262','#3e8351','#555d69'];
function performanceRegistry(){return data?.return_comparison?.instruments||[];}
function performanceChosen(){return performanceRegistry().filter(i=>performanceState.selected.has(i.instrument_id));}
function performanceReset(){
 const all=performanceChosen();if(!all.length)return;
 document.getElementById('performance-from').value=all.map(i=>i.start).sort().at(-1);
 document.getElementById('performance-to').value=all.map(i=>i.end).sort()[0];
}
function renderPerformanceResearch(){
 const registry=data.return_comparison||{instruments:[],errors:['Keine Vergleichsquelle vorhanden.']};
 if(!performanceState.initialized){registry.instruments.filter(i=>i.instrument_type==='strategy').forEach(i=>performanceState.selected.add(i.instrument_id));if(!performanceState.selected.size&&registry.instruments.length)performanceState.selected.add(registry.instruments[0].instrument_id);performanceState.initialized=true;performanceReset();}
 document.getElementById('performance-errors').textContent=(registry.errors||[]).join(' · ');
 renderHistoricalControls();renderInstrumentSearch();renderResearchStatus();renderPerformanceComparison();
}
function renderInstrumentSearch(){
 const query=document.getElementById('instrument-search').value.trim().toLowerCase();
 const matches=performanceRegistry().filter(i=>!query||(i.name+' '+i.instrument_id).toLowerCase().includes(query)||Object.values(i.identifiers).flat().some(v=>v.toLowerCase()===query));
 document.getElementById('performance-instruments').innerHTML=matches.length?matches.map(i=>`<label><input type="checkbox" data-instrument="${E(i.instrument_id)}" ${performanceState.selected.has(i.instrument_id)?'checked':''}> ${E(i.name)}<small>${E(Object.entries(i.identifiers).map(([k,v])=>k+': '+v.join(', ')).join(' · '))}<br>${E(i.share_class)} · ${E(i.exchange)} · ${E(i.currency)} · ${E(i.start)}–${E(i.end)}</small></label>`).join(''):'<p class="small">Kein belegter Treffer im lokalen Instrumentenstamm. Anbieter oder geprüften CSV-Import bereitstellen.</p>';
 document.querySelectorAll('[data-instrument]').forEach(input=>input.onchange=()=>{if(input.checked)performanceState.selected.add(input.dataset.instrument);else performanceState.selected.delete(input.dataset.instrument);performanceReset();renderPerformanceComparison();});
}
function comparisonView(instruments,start,end){
 if(!instruments.length)throw Error('Keine geprüfte Vergleichsreihe ausgewählt.');
 if(!start||!end||start>end)throw Error('Gültigen Von-/Bis-Zeitraum auswählen.');
 for(const key of ['currency','dividend_tax_basis','reinvestment_rule','valuation_convention'])if(new Set(instruments.map(i=>i[key])).size!==1)throw Error('Nicht vergleichbare Definition: '+key);
 const maps=instruments.map(i=>new Map(i.points.filter(p=>p.date>=start&&p.date<=end).map(p=>[p.date,p])));
 const dates=[...maps[0].keys()].filter(d=>maps.every(m=>m.has(d))).sort();
 if(dates.length<2)throw Error('Weniger als zwei gemeinsame belegte Bewertungstage; kein Auffüllen von Kurslücken.');
 for(const day of dates)if(new Set(maps.map(m=>m.get(day).valued_at)).size!==1)throw Error('Unterschiedliche Bewertungszeitpunkte; keine belegte gemeinsame Schlussbewertung.');
 const metric=document.getElementById('performance-metric').value;
 const series=instruments.map((instrument,i)=>{
  const values=dates.map(d=>maps[i].get(d).value),base=values[0];let peak=base;
  const points=values.map((v,j)=>{peak=Math.max(peak,v);return {date:dates[j],value:v,index100:v/base*100,return:v/base-1,drawdown:v/peak-1};});
  const years=(Date.parse(dates.at(-1))-Date.parse(dates[0]))/86400000/365.25;
  return {instrument,points,color:performanceColors[i%performanceColors.length],metrics:{total:values.at(-1)/base-1,cagr:Math.pow(values.at(-1)/base,1/years)-1,drawdown:Math.min(...points.map(p=>p.drawdown))}};
 });
 const allDates=new Set(maps.flatMap(m=>[...m.keys()]));
 return {dates,series,metric,excluded:allDates.size-dates.length,start:dates[0],end:dates.at(-1)};
}
function renderPerformanceComparison(){
 const status=document.getElementById('performance-range'),svg=document.getElementById('performance-svg');
 try{
  const view=comparisonView(performanceChosen(),document.getElementById('performance-from').value,document.getElementById('performance-to').value);performanceState.view=view;
  status.textContent=`Gemeinsamer Zeitraum ${view.start}–${view.end} · ${view.dates.length} gemeinsame Bewertungen · ${view.excluded} nicht gemeinsame Kalendertage ausgeschlossen (keine Nullrenditen). Alle Reihen am ${view.start} neu auf 100 normiert.`;
  document.getElementById('historical-weight-date').value=performanceState.holdingsDate&&view.dates.includes(performanceState.holdingsDate)?performanceState.holdingsDate:view.end;
  drawPerformance(view);renderHistoricalWeights();
  document.getElementById('performance-metrics').innerHTML='<thead><tr><th>Reihe</th><th>Rendite</th><th>CAGR</th><th>Max. Drawdown</th><th>Bewertungen</th></tr></thead><tbody>'+view.series.map(s=>`<tr><td>${E(s.instrument.name)}</td><td>${pct(s.metrics.total)}</td><td>${pct(s.metrics.cagr)}</td><td>${pct(s.metrics.drawdown)}</td><td>${num(view.dates.length)}</td></tr>`).join('')+'</tbody>';
  document.getElementById('performance-definitions').innerHTML=view.series.map(s=>`<div class="kv"><strong>${E(s.instrument.name)}</strong></div><pre>${E(JSON.stringify(Object.fromEntries(Object.entries(s.instrument).filter(([key])=>!['points','weight_history'].includes(key))),null,2))}</pre>`).join('');
 }catch(error){performanceState.view=null;status.textContent=error.message;svg.innerHTML='';document.getElementById('performance-hover').textContent='';document.getElementById('performance-metrics').innerHTML='';document.getElementById('performance-definitions').innerHTML='';document.getElementById('historical-weights').innerHTML='';document.getElementById('historical-weight-note').textContent='Keine belegte historische Gewichtsreihe für eine ausgeführte Simulation vorhanden.';}
}
function drawPerformance(view){
 const svg=document.getElementById('performance-svg'),W=1000,H=360,L=70,R=20,T=20,B=45;
 let lo=Infinity,hi=-Infinity;for(const series of view.series)for(const point of series.points){lo=Math.min(lo,point[view.metric]);hi=Math.max(hi,point[view.metric]);}if(lo===hi){lo-=1;hi+=1;}
 const x=i=>L+i/(view.dates.length-1)*(W-L-R),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 view.plot={L,R,T,B,W,H,x,y};
 const format=v=>view.metric==='index100'?v.toFixed(1):(v*100).toFixed(1)+' %';
 const grid=Array.from({length:5},(_,i)=>{const value=lo+(hi-lo)*i/4;return `<line x1="${L}" x2="${W-R}" y1="${y(value)}" y2="${y(value)}" stroke="#e2e8ec"/><text x="${L-8}" y="${y(value)+4}" text-anchor="end" font-size="12" fill="#61717d">${format(value)}</text>`;}).join('');
 const paths=view.series.map(s=>`<path d="${s.points.map((p,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(p[view.metric]).toFixed(2)).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2"><title>${E(s.instrument.name)}</title></path>`).join('');
 svg.innerHTML=`<title>Total Return ${view.start} bis ${view.end}</title>${grid}${paths}<text x="${L}" y="${H-10}" font-size="12">${view.start}</text><text x="${W-R}" y="${H-10}" text-anchor="end" font-size="12">${view.end}</text><line id="performance-cursor" x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" stroke="#172c3a" stroke-dasharray="4 3" visibility="hidden"/>`;
}
function performanceCursor(event){
 const view=performanceState.view;if(!view)return;
 const rect=document.getElementById('performance-svg').getBoundingClientRect(),px=(event.clientX-rect.left)/rect.width*1000;
 const index=Math.max(0,Math.min(view.dates.length-1,Math.round((px-view.plot.L)/(1000-view.plot.L-view.plot.R)*(view.dates.length-1))));
 const cursor=document.getElementById('performance-cursor');cursor.setAttribute('x1',view.plot.x(index));cursor.setAttribute('x2',view.plot.x(index));cursor.setAttribute('visibility','visible');
 document.getElementById('performance-hover').innerHTML=`<strong>${E(view.dates[index])}</strong>`+view.series.map(s=>`<span style="color:${s.color}">${E(s.instrument.name)}: ${view.metric==='index100'?s.points[index].index100.toFixed(2):pct(s.points[index][view.metric])}</span>`).join('');
 performanceState.holdingsDate=view.dates[index];document.getElementById('historical-weight-date').value=view.dates[index];document.getElementById('historical-asof').value=view.dates[index];renderHistoricalWeights();
}
function historicalInstrument(){return performanceRegistry().find(i=>i.instrument_id===document.getElementById('historical-series').value)||null;}
function renderHistoricalControls(){
 const select=document.getElementById('historical-series'),chosen=select.value;
 select.innerHTML='<option value="">Aktuelles Depot / Diagnosearchiv</option>'+performanceRegistry().filter(i=>i.instrument_type==='strategy'&&(i.weight_history||[]).length).map(i=>`<option value="${E(i.instrument_id)}">${E(i.name)}</option>`).join('');
 select.value=chosen;
}
function historicalPositions(){
 const instrument=historicalInstrument(),day=document.getElementById('historical-asof').value;
 return (instrument?.weight_history||[]).find(s=>s.date===day)?.positions||[];
}
function historicalStockRows(){
 return historicalPositions().map(p=>({...p,symbol:p.security_id,name:p.name||p.security_id,sector:sectors[p.sector]||p.sector||'unbekannt',target:p.target_weight,actual:p.actual_weight,delta:finite(p.actual_weight)&&finite(p.benchmark_weight)?p.actual_weight-p.benchmark_weight:null,forecast:null}))
 .sort((a,b)=>(b.benchmark_weight??-Infinity)-(a.benchmark_weight??-Infinity)||a.symbol.localeCompare(b.symbol));
}
function renderHistoricalStockView(){
 const instrument=historicalInstrument(),day=document.getElementById('historical-asof').value,all=filteredRows(),limit=25;state.page=Math.min(state.page,Math.max(0,Math.ceil(all.length/limit)-1));const visible=all.slice(state.page*limit,(state.page+1)*limit);
 document.getElementById('stocks-title').textContent='Historische Ziel- und Ist-Gewichte';
 document.getElementById('stocks-sub').textContent=`${instrument.name} · ${day||'Datum fehlt'} · keine heutigen Benchmarkgewichte, keine Renormierung. Live-Bestand bleibt gesondert; Diagnosegewichte sind keine Ausführungen.`;
 document.getElementById('stock-table').innerHTML='<thead><tr><th>Wertpapier</th><th>Sektor damals</th><th>Benchmark / SPY-Proxy* ↓</th><th>Modell-Ziel</th><th>Simulierter Bestand</th><th>Ist − Benchmark</th></tr></thead><tbody>'+visible.map(r=>`<tr><td class="${r.target>1e-9?'ml-selected':''}">${E(r.name)}<br>${E(r.symbol)}</td><td>${E(r.sector)}</td><td>${pct(r.benchmark_weight)}</td><td>${pct(r.target)}</td><td>${pct(r.actual)}</td><td>${pp(r.delta)}</td></tr>`).join('')+'</tbody>';
 document.getElementById('stocks-sub').textContent+=` ${all.length} historische Wertpapiere; fehlt ein Benchmarkgewicht, bleibt es offen.`;
 document.getElementById('page-info').textContent=all.length?`${state.page*limit+1}–${Math.min(all.length,(state.page+1)*limit)} von ${all.length} historischen Aktien`:'0 historische Aktien';
 document.getElementById('previous').disabled=state.page===0;document.getElementById('next').disabled=(state.page+1)*limit>=all.length;
 document.getElementById('view-msg').textContent='Benchmarkgewichte benötigen datierte Originalquellen; SPY bleibt als Proxy beschriftet. Fehlende Gewichte sind keine Nullgewichte. Tatsächliche Live-Bestände bleiben in der Depotansicht.';
}
function renderHistoricalSectorView(){
 const source=historicalStockRows(),groups=new Map();
 for(const row of source){if(!groups.has(row.sector))groups.set(row.sector,[]);groups.get(row.sector).push(row);}
 const rows=[...groups].map(([sector,positions])=>({sector,...Object.fromEntries(['benchmark_weight','target','actual'].map(key=>[key,positions.every(p=>finite(p[key]))?positions.reduce((s,p)=>s+p[key],0):null]))}));
 rows.sort((a,b)=>(b.benchmark_weight??-Infinity)-(a.benchmark_weight??-Infinity)||a.sector.localeCompare(b.sector));
 document.getElementById('sector-full-chart').innerHTML='';document.getElementById('sector-full-legend').innerHTML='';
 document.getElementById('sector-table').innerHTML='<thead><tr><th>Sektor damals</th><th>Benchmark / SPY-Proxy*</th><th>Modell-Ziel</th><th>Simulierter Bestand</th><th>Aktiv</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${E(r.sector)}</td><td>${pct(r.benchmark_weight)}</td><td>${pct(r.target)}</td><td>${pct(r.actual)}</td><td>${finite(r.actual)&&finite(r.benchmark_weight)?pp(r.actual-r.benchmark_weight):'–'}</td></tr>`).join('')+'</tbody>';
 document.getElementById('sector-note').textContent=`Historische Sicht ${document.getElementById('historical-asof').value}; ${source.length} Wertpapiere; keine Renormierung. Ohne vollständige Benchmarkquellen keine Sektor-Abweichung.`;
}
function renderHistoricalWeights(){
 const day=document.getElementById('historical-weight-date').value;performanceState.holdingsDate=day;
 const series=performanceChosen().filter(s=>s.instrument_type==='strategy');
 const snapshots=series.flatMap(s=>(s.weight_history||[]).filter(w=>w.date===day).map(w=>({...w,strategy:s.name})));
 const note=document.getElementById('historical-weight-note');
 if(!snapshots.length){note.textContent=`${day||'Kein Datum'}: keine belegten simulierten Bestände / historischen Benchmarkgewichte. Diagnosegewichte werden nicht eingesetzt. Live-Depot: siehe gesonderte Depotansicht.`;document.getElementById('historical-weights').innerHTML='';return;}
 const rows=snapshots.flatMap(s=>s.positions.map(p=>({...p,strategy:s.strategy}))).sort((a,b)=>(b.benchmark_weight??-Infinity)-(a.benchmark_weight??-Infinity)||a.security_id.localeCompare(b.security_id));
 note.textContent=`Exakte Bewertung ${day}; ohne belegtes Benchmarkgewicht bleibt die Spalte leer. Sektoren stammen aus der historischen Gewichtsreihe.`;
 document.getElementById('historical-weights').innerHTML='<thead><tr><th>Strategie</th><th>Wertpapier</th><th>Sektor</th><th>Benchmark / SPY-Proxy*</th><th>Modell-Ziel</th><th>Simulierter Bestand</th><th>Aktiv</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${E(r.strategy)}</td><td>${E(r.security_id)}</td><td>${E(r.sector||'unbekannt')}</td><td>${pct(r.benchmark_weight)}</td><td>${pct(r.target_weight)}</td><td>${pct(r.actual_weight)}</td><td>${finite(r.benchmark_weight)&&finite(r.actual_weight)?pp(r.actual_weight-r.benchmark_weight):'–'}</td></tr>`).join('')+'</tbody>';
}
function renderResearchStatus(){
 const packages=data.target_portfolio_packages||[];
 document.getElementById('target-package-list').innerHTML=packages.length?packages.map(p=>`<p><strong>${E(p.cycle_key||'Unbekannt')} · ${E(p.package_kind||p.integrity)}</strong> · ${p.handoff_allowed?'Zur Übergabe geprüft':'NICHT zur aktuellen Übergabe freigegeben'}<br>${E((p.handoff_blockers||[p.error]).filter(Boolean).join(' · '))}<br>${(p.downloads||[]).map(name=>`<a download href="/api/target-download?package=${encodeURIComponent(p.package_key)}&name=${encodeURIComponent(name)}">${E(name)}</a>`).join(' · ')}</p>`).join(''):'Keine abgeschlossenen Zielportfolio-Pakete für dieses Quartal. Numerische Diagnose und alte Archive sind keine aktuellen Übergabepakete.';
 const diagnostic=data.numeric_q2_diagnostic,diagnosticElement=document.getElementById('numeric-q2-diagnostic');
 if(diagnostic?.available){
  diagnosticElement.innerHTML=`<p><strong>NICHT AUSFÜHRBAR · KEIN BELEGTER HISTORISCHER Q2-LAUF</strong><br>${E(diagnostic.forecast_count)} Prognosen, ${E(diagnostic.selected_count)} ausgewählte Titel. Heutige Cache-Werte für Q2-Berichtsperioden; historische Trainingsversionen, Renditeziele und Kursbasis sind nicht zertifiziert.</p><p>${E(diagnostic.training_availability)}</p><table><thead><tr><th>Strategie</th><th>Numerisches Ergebnis</th><th>Gewichte</th><th>Fehler</th></tr></thead><tbody>${Object.entries(diagnostic.strategy_results).map(([name,r])=>`<tr><td>${E(name==='mean_variance'?'Rendite/Risiko – Max-Sharpe':name)}</td><td>${E(r.status)}</td><td>${E(Object.keys(r.weights).length)}</td><td>${E(r.error||'Nur Diagnose; nicht zur Übergabe freigegeben')}</td></tr>`).join('')}</tbody></table><p><a href="/api/diagnostic-download?name=forecasts_UNVERIFIED.csv" download>Diagnoseprognosen (CSV)</a> · <a href="/api/diagnostic-download?name=diagnostic.json" download>Vollständige Diagnose / Trainingszeiträume (JSON)</a></p>`;
 }else diagnosticElement.textContent=diagnostic?.error||'Für dieses Quartal keine neue numerische Diagnose.';
 if(diagnostic?.available&&diagnostic.original_reference_policy)diagnosticElement.innerHTML+=`<p><strong>Original-Referenzdaten-Annahme · keine PIT-Zertifizierung</strong><br>${E(diagnostic.original_reference_rows_checked)} tatsächlich verwendete Trainingszeilen wurden einschließlich aller 20 Merkmale und Renditeziel mit den bytegleichen Originaldateien abgeglichen. Die ausdrücklich akzeptierte Annahme ${E(diagnostic.original_reference_policy.version)} gilt nur für diese Originaldateien, nicht für SEC-Fortsetzungen, Indexmitgliedschaften oder unbekannte Zieldefinitionen. Der Commit-Zeitstempel von 2018 beweist keinen damaligen öffentlichen Upload. Historische Validierung bleibt NOT_VALIDATED.<br><a download href="/api/diagnostic-download?name=original_reference_rows.csv">Originalzeilen / Herkunftsabgleich (CSV)</a></p>`;
 if(diagnostic?.available)diagnosticElement.innerHTML+=`<p>Separate Diagnose-Dateien, keine ISIN-geprüften Übergabeportfolios: ${(diagnostic.downloads||[]).filter(name=>name.includes('_DIAGNOSTIC_')).map(name=>`<a download href="/api/diagnostic-download?name=${encodeURIComponent(name)}">${E(name)}</a>`).join(' · ')}</p>`;
 const progress=data.historical_source_progress;
 const note=document.getElementById('historical-source-progress');
 note.textContent=progress?.available?`${progress.as_of} · Öffentliche Quellen: ${progress.cover_identity_matches}/${progress.expected_securities} historische Q2-Deckblattidentitäten, ${progress.original_value_entity_matches} Originalwert-/Emittentenabgleiche, ${progress.market_sources_acquired} Kursquellen. ${progress.model_run_status}. Historische Validierung: ${progress.historical_validation}. ${progress.remaining_blockers.join(' · ')}. Die folgende Tabelle ist der separat eingefrorene frühere Auditstand; eine Vergleichskurve ist kein Strategielauf.`:(progress?.error||'');
 const research=data.fixed_lag_research;
 if(!research?.available){document.getElementById('fixed-lag-summary').textContent=research?.error||'Keine historische Forschungsauswertung vorhanden.';return;}
 const handoff=research.handoff;
 document.getElementById('fixed-lag-summary').textContent=`${research.run_id} · ${research.historical_validation} · letzter archivierter Modelltermin ${handoff.last_archived_model_date}, neue Fortsetzung ab ${handoff.first_new_execution}: ${handoff.new_decision_count} Entscheidungen. ${research.model_run_status}. Performance: ${research.performance_status}. Paper-Endwerte werden nicht an eine neue Kurve angehängt.`;
 document.getElementById('fixed-lag-quarters').innerHTML='<thead><tr><th>Quartal</th><th>Informationsschluss UTC</th><th>Ausführung</th><th>Status</th><th>Konkrete Blocker</th></tr></thead><tbody>'+research.quarters.map(q=>`<tr><td>${E(q.cycle_key)}</td><td>${E(q.information_cutoff)}</td><td>${E(q.execution_date)}</td><td>${E(q.status)}</td><td>${E(q.blockers.join(' · '))}</td></tr>`).join('')+'</tbody>';
 document.getElementById('fixed-lag-rules').innerHTML='<thead><tr><th>Paper</th><th>Bisher</th><th>Fortsetzung</th><th>Abweichung</th></tr></thead><tbody>'+research.rule_comparison.map(r=>`<tr><td>${E(r.paper)}</td><td>${E(r.current)}</td><td>${E(r.continuation)}</td><td>${E(r.deviation)}</td></tr>`).join('')+'</tbody>';
}
function performancePreset(range){
 const chosen=performanceChosen();if(!chosen.length)return;
 const end=chosen.map(i=>i.end).sort()[0],start=chosen.map(i=>i.start).sort().at(-1),day=new Date(end+'T12:00:00Z');
 if(range==='YTD'){day.setUTCMonth(0,1);}else if(range!=='Alles'){const months={'1M':1,'3M':3,'6M':6,'1J':12,'3J':36,'5J':60}[range];day.setUTCMonth(day.getUTCMonth()-months);}
 document.getElementById('performance-from').value=range==='Alles'?start:[start,day.toISOString().slice(0,10)].sort().at(-1);document.getElementById('performance-to').value=end;renderPerformanceComparison();
}
document.getElementById('instrument-search').oninput=renderInstrumentSearch;
['performance-from','performance-to','performance-metric'].forEach(id=>document.getElementById(id).onchange=renderPerformanceComparison);
document.getElementById('historical-weight-date').onchange=()=>{document.getElementById('historical-asof').value=document.getElementById('historical-weight-date').value;renderHistoricalWeights();};
document.getElementById('historical-series').onchange=()=>{state.page=0;const instrument=historicalInstrument();if(instrument&&!document.getElementById('historical-asof').value)document.getElementById('historical-asof').value=instrument.end;renderContent();};
document.getElementById('historical-asof').onchange=()=>{performanceState.holdingsDate=document.getElementById('historical-asof').value;state.page=0;renderContent();};
document.querySelectorAll('[data-range]').forEach(button=>button.onclick=()=>performancePreset(button.dataset.range));
document.getElementById('performance-quarter').onclick=()=>{
 const cycle=data.dashboard_research?.cycle_key||new URLSearchParams(location.search).get('cycle_key');
 if(!cycle)return;
 const schedule=data.fixed_lag_research?.quarters?.find(q=>q.cycle_key===cycle);
 if(!schedule){document.getElementById('performance-range').textContent='Kein belegter Quartalsplan für diese Auswahl.';return;}
 const next=data.fixed_lag_research.quarters.find(q=>q.cycle_key>cycle);
 document.getElementById('performance-from').value=schedule.execution_date;
 document.getElementById('performance-to').value=next?new Date(Date.parse(next.execution_date)-86400000).toISOString().slice(0,10):performanceChosen().map(i=>i.end).sort()[0]||schedule.execution_date;
 renderPerformanceComparison();
};
document.getElementById('performance-export').onclick=()=>{
 const view=performanceState.view;if(!view)return;
 const rows=[['date','instrument_id','value','index100','cumulative_return','drawdown','valuation_timestamp']];
 view.series.forEach(s=>s.points.forEach(p=>rows.push([p.date,s.instrument.instrument_id,p.value,p.index100,p.return,p.drawdown,s.instrument.points.find(v=>v.date===p.date).valued_at])));
 const csv=rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\r\n'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`comparison-${view.start}-${view.end}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
const performanceSvg=document.getElementById('performance-svg');
performanceSvg.addEventListener('pointermove',event=>{
 if(performanceState.drag&&performanceState.view){const drag=performanceState.drag,span=drag.end-drag.start,delta=(event.clientX-drag.x)/performanceSvg.getBoundingClientRect().width*span;document.getElementById('performance-from').value=new Date(drag.start-delta).toISOString().slice(0,10);document.getElementById('performance-to').value=new Date(drag.end-delta).toISOString().slice(0,10);renderPerformanceComparison();}
 performanceCursor(event);
});
performanceSvg.addEventListener('pointerdown',event=>{if(!performanceState.view)return;performanceState.drag={x:event.clientX,start:Date.parse(document.getElementById('performance-from').value),end:Date.parse(document.getElementById('performance-to').value)};performanceSvg.setPointerCapture(event.pointerId);});
performanceSvg.addEventListener('pointerup',()=>performanceState.drag=null);
performanceSvg.addEventListener('pointercancel',()=>performanceState.drag=null);
performanceSvg.addEventListener('wheel',event=>{
 if(!performanceState.view)return;event.preventDefault();
 const start=Date.parse(document.getElementById('performance-from').value),end=Date.parse(document.getElementById('performance-to').value),span=end-start;if(span<86400000)return;
 const rect=performanceSvg.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),center=start+span*fraction,factor=event.deltaY<0?.8:1.25;
 document.getElementById('performance-from').value=new Date(center-(center-start)*factor).toISOString().slice(0,10);
 document.getElementById('performance-to').value=new Date(center+(end-center)*factor).toISOString().slice(0,10);renderPerformanceComparison();
},{passive:false});
if(location.hash==='#performance')go('performance');
"""


def extend_html(html: str) -> str:
    changes = [
        ('<label for="strategy">Strategie</label>',
         '<label for="historical-series">Gewichtsquelle</label><select id="historical-series">'
         '<option value="">Aktuelles Depot / Diagnosearchiv</option></select>'
         '<label for="historical-asof">Historisches Datum</label><input type="date" id="historical-asof">'
         '<label for="strategy">Strategie</label>'),
        ('function rows(){', "function rows(){if(['stocks','sectors'].includes(state.tab)&&historicalInstrument())return historicalStockRows();"),
        ('function renderStocks(){', 'function renderStocks(){if(historicalInstrument())return renderHistoricalStockView();'),
        ('function renderSectors(){', 'function renderSectors(){if(historicalInstrument())return renderHistoricalSectorView();'),
        ("if(state.tab==='overview')renderOverview();", "renderHistoricalControls();if(state.tab==='overview')renderOverview();"),
        ('<section id="panel-publications"', PANEL + '<section id="panel-publications"'),
        ('<button role="tab" data-tab="publications"',
         '<button role="tab" data-tab="performance" aria-selected="false">Performance · Historie</button>'
         '<button role="tab" data-tab="publications"'),
        ("if(state.tab==='publications')renderPublications();",
         "if(state.tab==='performance')renderPerformanceResearch();if(state.tab==='publications')renderPublications();"),
        ("function go(tab){", "function go(tab){if(tab==='performance')history.replaceState(null,'','#performance');"
         "else if(location.hash==='#performance')history.replaceState(null,'',location.pathname+location.search);"),
        ("</style>", STYLE + "</style>"), ("</script>", SCRIPT + "</script>"),
    ]
    for old, new in changes:
        if html.count(old) != 1:
            raise RuntimeError(f"Performance template anchor missing/ambiguous: {old}")
        html = html.replace(old, new, 1)
    return html
