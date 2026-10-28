import { useState } from "react"
import type { FormEvent } from "react"
import "./App.css"

type Prediction = {
  pitch_type: string
  probability: number
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

  async function handlePredict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setLoading(true)
    setError("")
    setPredictions([])

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          balls,
          strikes,
          stand,
          inning,
          outs_when_up: outs,
          // Match the terminal predictor's handling of a new plate appearance.
          previous_pitch: isFirstPitch ? "START" : previousPitch,
        }),
      })

      if (!response.ok) {
        throw new Error(`Prediction failed (HTTP ${response.status}).`)
      }

      const data = await response.json()
      setPredictions(data.probabilities)
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Could not get a prediction."
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

      {error && <p role="alert">{error} Check that FastAPI is running.</p>}

      <section aria-live="polite">
        {predictions.length > 0 && (
          <>
            <h2>Estimated probabilities</h2>
            <p>Results for your last submitted situation.</p>
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