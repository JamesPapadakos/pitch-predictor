import { useEffect, useState } from "react"
import type { SubmitEvent } from "react"
import "./App.css"

type Prediction = {
  pitch_type: string
  probability: number
}

type Situation = {
  balls: number
  strikes: number
  stand: string
  inning: number
  outs_when_up: number
  previous_pitch: string
}

const pitchNames: Record<string, string> = {
  FF: "Four-seam fastball",
  SL: "Slider",
  KC: "Knuckle curve",
  CH: "Changeup",
  FC: "Cutter",
  SI: "Sinker",
  ST: "Sweeper",
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "/api"
).replace(/\/+$/, "")

function ordinal(value: number): string {
  const lastTwo = value % 100

  if (lastTwo >= 11 && lastTwo <= 13) {
    return `${value}th`
  }

  const suffix = { 1: "st", 2: "nd", 3: "rd" }[value % 10] ?? "th"
  return `${value}${suffix}`
}

function App() {
  const [balls, setBalls] = useState(1)
  const [strikes, setStrikes] = useState(2)
  const [stand, setStand] = useState("L")
  const [inning, setInning] = useState(6)
  const [outs, setOuts] = useState(1)
  const [previousPitch, setPreviousPitch] = useState("FF")

  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const isFirstPitch = balls === 0 && strikes === 0
  const topPrediction = predictions[0]

  const [submittedSituation, setSubmittedSituation] =
    useState<Situation | null>(null)

  const [showWakeupMessage, setShowWakeupMessage] = useState(false)

useEffect(() => {
  if (!loading) {
    return
  }

  // Explain longer waits without showing a server message on every request.
  const timer = window.setTimeout(() => {
    setShowWakeupMessage(true)
  }, 5000)

  return () => window.clearTimeout(timer)
}, [loading])

async function handlePredict(event: SubmitEvent<HTMLFormElement>) {
  event.preventDefault()

  // Keep the displayed context identical to the request that produced it.
  const situation: Situation = {
    balls,
    strikes,
    stand,
    inning,
    outs_when_up: outs,
    previous_pitch: isFirstPitch ? "START" : previousPitch,
  }
  setShowWakeupMessage(false)
  setLoading(true)
  setError("")
  setPredictions([])
  setSubmittedSituation(null)

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(situation),
    })

    if (!response.ok) {
      throw new Error(`Prediction failed (HTTP ${response.status}).`)
    }

    const data = await response.json()
    setPredictions(data.probabilities)
    setSubmittedSituation(situation)
  } catch (error) {
    setError(
      error instanceof Error
        ? error.message
        : "Could not get a prediction."
    )
  } finally {
    setLoading(false)
  }
}

  return (
    <main>
      <h1>Pitch Predictor</h1>
      <p>Estimate Dylan Cease’s next pitch type.</p>
      <form onSubmit={handlePredict}>
        <fieldset disabled={loading}>
          <label>
            Balls
            <select value={balls} onChange={(e) => setBalls(Number(e.target.value))}>
              {[0, 1, 2, 3].map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </label>

          <label>
            Strikes
            <select value={strikes} onChange={(e) => setStrikes(Number(e.target.value))}>
              {[0, 1, 2].map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </label>

          <label>
            Batter handedness
            <select value={stand} onChange={(e) => setStand(e.target.value)}>
              <option value="L">Left-handed</option>
              <option value="R">Right-handed</option>
            </select>
          </label>

          <label>
            Inning
            <input
              type="number"
              min="1"
              step="1"
              required
              value={Number.isNaN(inning) ? "" : inning}
              onChange={(e) => setInning(e.target.valueAsNumber)}
            />
          </label>

          <label>
            Outs
            <select value={outs} onChange={(e) => setOuts(Number(e.target.value))}>
              {[0, 1, 2].map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </label>

          <label>
            Previous pitch
            <select
              value={isFirstPitch ? "START" : previousPitch}
              disabled={isFirstPitch}
              onChange={(e) => setPreviousPitch(e.target.value)}
            >
              <option value="START" disabled>Start of plate appearance</option>
              {Object.entries(pitchNames).map(([code, name]) => (
                <option key={code} value={code}>{name}</option>
              ))}
              <option value="UNKNOWN">Unknown</option>
            </select>
          </label>

          <button type="submit">
            {loading ? "Predicting…" : "Predict"}
          </button>
        </fieldset>
      </form>
      {loading && (
        <div className="loading-message" role="status">
          <span className="loading-spinner" aria-hidden="true" />

          <div>
            <p>Generating prediction…</p>

            {showWakeupMessage && (
              <p className="loading-detail">
                The server may be waking up. Your first prediction can
                take about a minute.
              </p>
            )}
          </div>
        </div>
      )}

      {error && <p role="alert">{error} Check that FastAPI is running.</p>}

      <section aria-live="polite">
        {predictions.length > 0 && (
          <>
            <h2>Estimated probabilities</h2>
            {submittedSituation && (
              <div className="situation-summary">
                <p>
                  <strong>
                    {submittedSituation.balls}–{submittedSituation.strikes} count
                  </strong>
                  {" · "}
                  {submittedSituation.stand === "L" ? "Left-handed" : "Right-handed"} batter
                  {" · "}
                  {ordinal(submittedSituation.inning)} inning
                  {" · "}
                  {submittedSituation.outs_when_up}{" "}
                  {submittedSituation.outs_when_up === 1 ? "out" : "outs"}
                </p>

                <p>
                  Previous pitch:{" "}
                  {submittedSituation.previous_pitch === "START"
                    ? "Start of plate appearance"
                    : submittedSituation.previous_pitch === "UNKNOWN"
                      ? "Unknown"
                      : pitchNames[submittedSituation.previous_pitch]
                        ?? submittedSituation.previous_pitch}
                </p>
              </div>
            )}
            {topPrediction && (
              <div className="top-pitch-card">
                <div>
                  <p className="top-pitch-label">Most likely pitch</p>
                  <h3>
                    {pitchNames[topPrediction.pitch_type]
                      ?? topPrediction.pitch_type}
                  </h3>
                </div>

                <div className="top-pitch-probability">
                  <strong>
                    {(topPrediction.probability * 100).toFixed(1)}%
                  </strong>
                  <span>Estimated probability</span>
                </div>
              </div>
            )}

            <h3>Pitch breakdown</h3>
            {predictions.map(({ pitch_type, probability }) => (
              <div className="prediction" key={pitch_type}>
                <div className="prediction-label">
                  <span>{pitchNames[pitch_type] ?? pitch_type}</span>
                  <strong>{(probability * 100).toFixed(1)}%</strong>
                </div>
                <progress
                  value={probability}
                  max={1}
                  aria-label={pitchNames[pitch_type] ?? pitch_type}
                />
              </div>
            ))}
            <p className="note">
              Estimates from a model trained on Dylan Cease’s 2022–2026 games with more weight on recent pitches.
              Probabilities are not guarantees.
            </p>
          </>
        )}
      </section>
    </main>
  )
}

export default App