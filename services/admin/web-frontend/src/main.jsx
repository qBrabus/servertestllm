import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'

const REFRESH_MS = 5000

function usePoller(url, refreshMs = REFRESH_MS) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let timer
    let controller
    let cancelled = false

    async function tick() {
      controller = new AbortController()
      try {
        const resp = await fetch(url, { signal: controller.signal })
        if (!resp.ok) {
          throw new Error(`${resp.status} ${resp.statusText}`)
        }
        const json = await resp.json()
        if (!cancelled) {
          setData(json)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err?.message ?? String(err))
        }
      } finally {
        if (!cancelled) {
          timer = setTimeout(tick, refreshMs)
        }
      }
    }

    tick()

    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
      if (controller) controller.abort()
    }
  }, [url, refreshMs])

  return { data, error }
}

function Card({ title, status, children, footer }) {
  return (
    <div style={{
      borderRadius: 16,
      border: '1px solid rgba(255,255,255,0.08)',
      padding: 20,
      background: 'linear-gradient(160deg, rgba(255,255,255,0.06), rgba(20,20,20,0.9))',
      color: '#f7f7f7',
      boxShadow: '0 18px 40px rgba(0,0,0,0.35)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 20 }}>{title}</h3>
        {status}
      </div>
      <div>{children}</div>
      {footer && <div style={{ marginTop: 16, fontSize: 13, opacity: 0.75 }}>{footer}</div>}
    </div>
  )
}

function StatusPill({ ok }) {
  const label = ok ? 'OPÉRATIONNEL' : 'HORS LIGNE'
  const background = ok ? 'rgba(76, 175, 80, 0.2)' : 'rgba(244, 67, 54, 0.2)'
  const color = ok ? '#81c784' : '#ef9a9a'
  return (
    <span style={{
      padding: '4px 12px',
      borderRadius: 999,
      fontWeight: 600,
      letterSpacing: 1,
      fontSize: 12,
      textTransform: 'uppercase',
      background,
      color
    }}>{label}</span>
  )
}

function ServiceCard({ service }) {
  const ok = service?.ok
  return (
    <Card
      title={service.name}
      status={<StatusPill ok={ok} />}
      footer={service.base_url}
    >
      <div style={{ fontSize: 14, lineHeight: 1.6 }}>
        <div>Catégorie : <strong>{service.category}</strong></div>
        {service.error && (
          <pre style={{
            marginTop: 12,
            background: 'rgba(244, 67, 54, 0.12)',
            padding: 12,
            borderRadius: 12,
            color: '#ffCDD2',
            whiteSpace: 'pre-wrap'
          }}>{service.error}</pre>
        )}
        {service.response && ok && (
          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: 'pointer' }}>Réponse /health</summary>
            <pre style={{
              marginTop: 8,
              background: 'rgba(255,255,255,0.05)',
              padding: 12,
              borderRadius: 12,
              whiteSpace: 'pre-wrap'
            }}>{JSON.stringify(service.response, null, 2)}</pre>
          </details>
        )}
        <div style={{ marginTop: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Points d'accès :</div>
          <ul style={{ paddingLeft: 20, margin: 0 }}>
            {service.public_endpoints?.map((ep) => (
              <li key={ep.url} style={{ marginBottom: 4 }}>
                <code>{ep.label}</code>
                <br />
                <span style={{ opacity: 0.75, fontSize: 12 }}>{ep.url}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Card>
  )
}

function GPUCard({ gpu }) {
  const memPct = useMemo(() => {
    if (!gpu.memory_total_mb) return 0
    return Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100)
  }, [gpu])

  return (
    <Card
      title={`${gpu.name} (#${gpu.index})`}
      status={<StatusPill ok={gpu.utilization_pct < 95} />}
      footer={`nvidia-smi • Température ${gpu.temperature_c}°C`}
    >
      <div style={{ fontSize: 14, lineHeight: 1.7 }}>
        <div>Utilisation GPU : <strong>{gpu.utilization_pct}%</strong></div>
        <div>VRAM : <strong>{gpu.memory_used_mb} / {gpu.memory_total_mb} MB ({memPct}%)</strong></div>
        <div style={{
          marginTop: 12,
          height: 10,
          background: 'rgba(255,255,255,0.1)',
          borderRadius: 999,
          overflow: 'hidden'
        }}>
          <div style={{
            height: '100%',
            width: `${memPct}%`,
            background: 'linear-gradient(90deg, #4caf50, #81c784)'
          }} />
        </div>
      </div>
    </Card>
  )
}

function ConfigPanel({ config }) {
  return (
    <Card title="Configuration" status={<span />}>
      <div style={{ fontSize: 14, lineHeight: 1.7 }}>
        <div>Répertoire modèles monté : <code>{config.models_root}</code></div>
        <div>Cache Hugging Face : <code>{config.huggingface_cache}</code></div>
        <div style={{ marginTop: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Services déclarés :</div>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {config.services?.map((svc) => (
              <li key={svc.id}>
                <strong>{svc.name}</strong> — {svc.base_url}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Card>
  )
}

function Section({ title, children, description }) {
  return (
    <section style={{ marginTop: 32 }}>
      <header style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, color: '#fafafa', fontSize: 26 }}>{title}</h2>
        {description && <p style={{ margin: '8px 0 0', color: 'rgba(255,255,255,0.65)' }}>{description}</p>}
      </header>
      {children}
    </section>
  )
}

function App() {
  const { data: servicesData, error: servicesError } = usePoller('/api/services')
  const { data: gpuData, error: gpuError } = usePoller('/api/gpus')
  const { data: configData } = usePoller('/api/config', 30000)

  return (
    <div style={{
      padding: '40px min(5vw, 64px)',
      fontFamily: 'Inter, system-ui, sans-serif',
      background: 'radial-gradient(circle at top, #1f2933, #050505 60%)',
      minHeight: '100vh'
    }}>
      <header style={{ marginBottom: 32 }}>
        <h1 style={{ color: '#fff', fontSize: 36, marginBottom: 8 }}>AI Stack — Pilotage</h1>
        <p style={{ color: 'rgba(255,255,255,0.7)', maxWidth: 720 }}>
          Supervision unifiée des services Qwen vLLM, NVIDIA Canary (ASR) et pyannote (diarisation).
          Surveillez l'état des APIs, la configuration déployée ainsi que l'utilisation GPU en temps réel.
        </p>
      </header>

      <Section title="Services" description="Suivi de l'état de santé et des points d'accès OpenAI/REST de chaque composant.">
        {servicesError && (
          <div style={{
            padding: 16,
            borderRadius: 12,
            background: 'rgba(244, 67, 54, 0.18)',
            color: '#ffcdd2',
            marginBottom: 16
          }}>
            Impossible de joindre l'API d'administration : {servicesError}
          </div>
        )}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: 20
        }}>
          {servicesData?.services?.map((svc) => (
            <ServiceCard key={svc.id} service={svc} />
          ))}
        </div>
      </Section>

      <Section title="GPU" description="Lecture directe de nvidia-smi depuis le conteneur admin.">
        {gpuError && (
          <div style={{
            padding: 16,
            borderRadius: 12,
            background: 'rgba(255, 193, 7, 0.15)',
            color: '#ffe082',
            marginBottom: 16
          }}>
            Attention : {gpuError}
          </div>
        )}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: 20
        }}>
          {gpuData?.gpus?.length
            ? gpuData.gpus.map((gpu) => <GPUCard key={gpu.index} gpu={gpu} />)
            : <Card title="Aucun GPU détecté" status={<StatusPill ok={false} />} footer="nvidia-smi">
                <div>Le runtime n'a pas exposé de GPU au conteneur admin.</div>
              </Card>}
        </div>
      </Section>

      {configData && (
        <Section title="Configuration" description="Variables d'environnement injectées et routage public.">
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: 20
          }}>
            <ConfigPanel config={configData} />
          </div>
        </Section>
      )}
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
