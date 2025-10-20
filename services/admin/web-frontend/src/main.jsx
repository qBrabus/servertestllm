import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

function useFetch(url, refreshMs=3000) {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => {
    let timer
    async function tick() {
      try {
        const r = await fetch(url)
        setData(await r.json())
        setErr(null)
      } catch (e) {
        setErr(String(e))
      } finally {
        timer = setTimeout(tick, refreshMs)
      }
    }
    tick()
    return () => clearTimeout(timer)
  }, [url, refreshMs])
  return { data, err }
}

function GPUCard({ g }) {
  const memPct = Math.round(100 * g.memory_used_mb / g.memory_total_mb)
  return (
    <div style={{border:'1px solid #333', padding:12, borderRadius:12, background:'#111', color:'#eee'}}>
      <div style={{fontSize:18, fontWeight:'bold'}}>{g.name} (#{g.index})</div>
      <div>Utilisation: {g.utilization_pct}% • Temp: {g.temperature_c}°C</div>
      <div>VRAM: {g.memory_used_mb} / {g.memory_total_mb} MB ({memPct}%)</div>
      <div style={{height:8, background:'#333', borderRadius:6, marginTop:6}}>
        <div style={{height:'100%', width:`${memPct}%`, background:'#4caf50', borderRadius:6}}/>
      </div>
    </div>
  )
}

function ServiceCard({ name, status }) {
  const ok = status?.ok
  return (
    <div style={{border:'1px solid #333', padding:12, borderRadius:12, background:'#111', color:'#eee'}}>
      <div style={{fontSize:18, fontWeight:'bold'}}>{name}</div>
      <div>Status: {ok ? 'OK' : 'Down'}</div>
      {!ok && <pre style={{whiteSpace:'pre-wrap'}}>{JSON.stringify(status, null, 2)}</pre>}
    </div>
  )
}

function App() {
  const { data: gpus } = useFetch('/api/gpus', 4000)
  const { data: services } = useFetch('/api/services', 4000)

  return (
    <div style={{padding:20, fontFamily:'system-ui, sans-serif', background:'#0a0a0a', minHeight:'100vh'}}>
      <h1 style={{color:'#fff'}}>LLM / ASR / Diar Admin</h1>

      <section>
        <h2 style={{color:'#ddd'}}>Services</h2>
        <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(260px,1fr))', gap:12}}>
          <ServiceCard name="vLLM (Qwen3-VL-30B-A3B)" status={services?.vllm} />
          <ServiceCard name="ASR (Canary 1B v2)" status={services?.asr} />
          <ServiceCard name="Diarization (pyannote community-1)" status={services?.diar} />
        </div>
      </section>

      <section>
        <h2 style={{color:'#ddd'}}>GPUs</h2>
        <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(300px, 1fr))', gap:12}}>
          {gpus?.gpus?.map(g => <GPUCard key={g.index} g={g} />)}
        </div>
      </section>

      <section style={{marginTop:30, color:'#ccc'}}>
        <h2>Endpoints</h2>
        <ul>
          <li>OpenAI LLM: <code>/v1/chat/completions</code> via vLLM</li>
          <li>ASR: <code>/v1/audio/transcriptions</code> via Canary</li>
          <li>Diarization: <code>/v1/audio/diarize</code> via pyannote</li>
        </ul>
      </section>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App/>)
