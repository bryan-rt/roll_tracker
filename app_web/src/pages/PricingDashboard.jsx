import { useState, useMemo } from 'react'

const TIERS = [
  { key: 'starter', label: 'Starter', price: 99, cameras: 1 },
  { key: 'standard', label: 'Standard', price: 199, cameras: 3 },
  { key: 'premium', label: 'Premium', price: 349, cameras: 6 },
]

const COST_ITEMS = [
  { key: 'cameras', label: 'Nest cameras (amortized/mo)', perGym: true, base: 8 },
  { key: 'compute', label: 'Compute & storage', perGym: true, base: 25 },
  { key: 'supabase', label: 'Supabase (Pro plan share)', perGym: false, flat: 25 },
  { key: 'support', label: 'Support & ops', perGym: false, flat: 200 },
]

const BILLING_CHANNELS = [
  { key: 'direct', label: 'Direct (Stripe)', cut: 0.029 },
  { key: 'appstore', label: 'App Store', cut: 0.30 },
]

function fmt(n) {
  if (Math.abs(n) >= 1000) return `$${(n / 1000).toFixed(1)}k`
  return `$${Math.round(n)}`
}

function pct(n) {
  return `${(n * 100).toFixed(1)}%`
}

export default function PricingDashboard() {
  const [tab, setTab] = useState('model')

  // Model inputs
  const [gymCount, setGymCount] = useState(20)
  const [athletesPerGym, setAthletesPerGym] = useState(80)
  const [clipsPerAthleteMonth, setClipsPerAthleteMonth] = useState(8)
  const [tierMix, setTierMix] = useState({ starter: 0.3, standard: 0.5, premium: 0.2 })
  const [billingChannel, setBillingChannel] = useState('direct')
  const [clipFeeEnabled, setClipFeeEnabled] = useState(false)
  const [clipFee, setClipFee] = useState(0.25)
  const [athleteSubEnabled, setAthleteSubEnabled] = useState(false)
  const [athleteSubPrice, setAthleteSubPrice] = useState(4.99)
  const [athleteSubAdoption, setAthleteSubAdoption] = useState(0.15)
  const [costToggles, setCostToggles] = useState(
    Object.fromEntries(COST_ITEMS.map(c => [c.key, true]))
  )
  const [notes, setNotes] = useState('')

  const channel = BILLING_CHANNELS.find(c => c.key === billingChannel)

  const metrics = useMemo(() => {
    // Revenue
    const gymRevenue = TIERS.reduce((sum, tier) => {
      const count = Math.round(gymCount * tierMix[tier.key])
      return sum + count * tier.price
    }, 0)

    const totalAthletes = gymCount * athletesPerGym
    const totalClips = totalAthletes * clipsPerAthleteMonth

    const clipRevenue = clipFeeEnabled ? totalClips * clipFee : 0
    const athleteSubRevenue = athleteSubEnabled
      ? totalAthletes * athleteSubAdoption * athleteSubPrice
      : 0

    const grossRevenue = gymRevenue + clipRevenue + athleteSubRevenue
    const processingFees = grossRevenue * channel.cut
    const netRevenue = grossRevenue - processingFees

    // Costs
    let totalCosts = 0
    const costBreakdown = COST_ITEMS.map(item => {
      if (!costToggles[item.key]) return { ...item, amount: 0 }
      const amount = item.perGym ? item.base * gymCount : item.flat
      totalCosts += amount
      return { ...item, amount }
    })

    const margin = netRevenue - totalCosts
    const marginPct = netRevenue > 0 ? margin / netRevenue : 0

    // Per-unit
    const revenuePerGym = gymCount > 0 ? netRevenue / gymCount : 0
    const costPerGym = gymCount > 0 ? totalCosts / gymCount : 0
    const marginPerGym = revenuePerGym - costPerGym
    const revenuePerAthlete = totalAthletes > 0 ? netRevenue / totalAthletes : 0

    // Fleet projection (12 months)
    const fleet = Array.from({ length: 12 }, (_, i) => {
      const month = i + 1
      const gyms = Math.round(gymCount * (1 + i * 0.08))
      const rev = TIERS.reduce((sum, tier) => {
        return sum + Math.round(gyms * tierMix[tier.key]) * tier.price
      }, 0)
      const athletes = gyms * athletesPerGym
      const clips = athletes * clipsPerAthleteMonth
      const cRev = clipFeeEnabled ? clips * clipFee : 0
      const aRev = athleteSubEnabled ? athletes * athleteSubAdoption * athleteSubPrice : 0
      const gross = rev + cRev + aRev
      const net = gross * (1 - channel.cut)
      let costs = 0
      COST_ITEMS.forEach(item => {
        if (!costToggles[item.key]) return
        costs += item.perGym ? item.base * gyms : item.flat
      })
      return { month, gyms, revenue: net, costs, margin: net - costs }
    })

    return {
      gymRevenue, clipRevenue, athleteSubRevenue,
      grossRevenue, processingFees, netRevenue,
      totalCosts, costBreakdown, margin, marginPct,
      revenuePerGym, costPerGym, marginPerGym, revenuePerAthlete,
      totalAthletes, totalClips, fleet,
    }
  }, [gymCount, athletesPerGym, clipsPerAthleteMonth, tierMix, billingChannel,
      clipFeeEnabled, clipFee, athleteSubEnabled, athleteSubPrice, athleteSubAdoption,
      costToggles, channel])

  // Sensitivity matrix
  const sensitivityData = useMemo(() => {
    const gymRange = [5, 10, 20, 40, 80, 150]
    const athleteRange = [30, 50, 80, 120, 200]
    return gymRange.map(g => {
      return athleteRange.map(a => {
        const rev = TIERS.reduce((sum, tier) => {
          return sum + Math.round(g * tierMix[tier.key]) * tier.price
        }, 0)
        const athletes = g * a
        const cRev = clipFeeEnabled ? athletes * clipsPerAthleteMonth * clipFee : 0
        const aRev = athleteSubEnabled ? athletes * athleteSubAdoption * athleteSubPrice : 0
        const net = (rev + cRev + aRev) * (1 - channel.cut)
        let costs = 0
        COST_ITEMS.forEach(item => {
          if (!costToggles[item.key]) return
          costs += item.perGym ? item.base * g : item.flat
        })
        return { gyms: g, athletes: a, margin: net - costs, marginPct: net > 0 ? (net - costs) / net : 0 }
      })
    })
  }, [tierMix, clipFeeEnabled, clipFee, athleteSubEnabled, athleteSubPrice,
      athleteSubAdoption, clipsPerAthleteMonth, costToggles, channel])

  const maxWaterfall = Math.max(metrics.netRevenue, metrics.grossRevenue, 1)

  return (
    <div style={styles.root}>
      <h2 style={styles.header}>Roll Tracker &middot; Admin &mdash; Business model simulator</h2>

      {/* Tabs */}
      <div style={styles.tabs}>
        {['model', 'unit', 'sensitivity', 'notes'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              ...styles.tab,
              ...(tab === t ? styles.tabActive : {}),
            }}
          >
            {t === 'model' ? 'Model' : t === 'unit' ? 'Unit Economics' : t === 'sensitivity' ? 'Sensitivity' : 'Notes'}
          </button>
        ))}
      </div>

      {/* Model Tab */}
      {tab === 'model' && (
        <div style={styles.content}>
          <div style={styles.row}>
            {/* Left column — inputs */}
            <div style={styles.col}>
              <h3>Revenue Streams</h3>

              <label style={styles.sliderLabel}>
                Gym count: <strong>{gymCount}</strong>
                <input type="range" min={1} max={200} value={gymCount}
                  onChange={e => setGymCount(+e.target.value)} style={styles.slider} />
              </label>

              <label style={styles.sliderLabel}>
                Athletes / gym: <strong>{athletesPerGym}</strong>
                <input type="range" min={10} max={300} value={athletesPerGym}
                  onChange={e => setAthletesPerGym(+e.target.value)} style={styles.slider} />
              </label>

              <label style={styles.sliderLabel}>
                Clips / athlete / mo: <strong>{clipsPerAthleteMonth}</strong>
                <input type="range" min={1} max={30} value={clipsPerAthleteMonth}
                  onChange={e => setClipsPerAthleteMonth(+e.target.value)} style={styles.slider} />
              </label>

              <h4>Tier Mix</h4>
              {TIERS.map(tier => (
                <label key={tier.key} style={styles.sliderLabel}>
                  {tier.label} ({fmt(tier.price)}/mo): <strong>{pct(tierMix[tier.key])}</strong>
                  <input type="range" min={0} max={100} value={Math.round(tierMix[tier.key] * 100)}
                    onChange={e => {
                      const val = +e.target.value / 100
                      setTierMix(prev => ({ ...prev, [tier.key]: val }))
                    }} style={styles.slider} />
                </label>
              ))}

              <h4>Billing Channel</h4>
              {BILLING_CHANNELS.map(ch => (
                <label key={ch.key} style={styles.radioLabel}>
                  <input type="radio" name="channel" value={ch.key}
                    checked={billingChannel === ch.key}
                    onChange={() => setBillingChannel(ch.key)} />
                  {ch.label} ({pct(ch.cut)} fee)
                </label>
              ))}

              <h4>Add-on Revenue</h4>
              <label style={styles.checkLabel}>
                <input type="checkbox" checked={clipFeeEnabled}
                  onChange={e => setClipFeeEnabled(e.target.checked)} />
                Per-clip fee: ${clipFee.toFixed(2)}
              </label>
              {clipFeeEnabled && (
                <input type="range" min={5} max={100} value={Math.round(clipFee * 100)}
                  onChange={e => setClipFee(+e.target.value / 100)} style={styles.slider} />
              )}

              <label style={styles.checkLabel}>
                <input type="checkbox" checked={athleteSubEnabled}
                  onChange={e => setAthleteSubEnabled(e.target.checked)} />
                Athlete subscription: ${athleteSubPrice.toFixed(2)}/mo
              </label>
              {athleteSubEnabled && (
                <>
                  <label style={styles.sliderLabel}>
                    Price: <strong>${athleteSubPrice.toFixed(2)}</strong>
                    <input type="range" min={100} max={1999} value={Math.round(athleteSubPrice * 100)}
                      onChange={e => setAthleteSubPrice(+e.target.value / 100)} style={styles.slider} />
                  </label>
                  <label style={styles.sliderLabel}>
                    Adoption: <strong>{pct(athleteSubAdoption)}</strong>
                    <input type="range" min={1} max={50} value={Math.round(athleteSubAdoption * 100)}
                      onChange={e => setAthleteSubAdoption(+e.target.value / 100)} style={styles.slider} />
                  </label>
                </>
              )}

              <h3 style={{ marginTop: '1.5rem' }}>Costs</h3>
              {COST_ITEMS.map(item => (
                <label key={item.key} style={styles.checkLabel}>
                  <input type="checkbox" checked={costToggles[item.key]}
                    onChange={e => setCostToggles(prev => ({ ...prev, [item.key]: e.target.checked }))} />
                  {item.label}: {item.perGym ? `${fmt(item.base)}/gym` : `${fmt(item.flat)} flat`}
                </label>
              ))}
            </div>

            {/* Right column — summary + waterfall */}
            <div style={styles.col}>
              <h3>Monthly Summary</h3>
              <table style={styles.table}>
                <tbody>
                  <tr><td>Gym subscription revenue</td><td style={styles.num}>{fmt(metrics.gymRevenue)}</td></tr>
                  {clipFeeEnabled && <tr><td>Clip fee revenue</td><td style={styles.num}>{fmt(metrics.clipRevenue)}</td></tr>}
                  {athleteSubEnabled && <tr><td>Athlete sub revenue</td><td style={styles.num}>{fmt(metrics.athleteSubRevenue)}</td></tr>}
                  <tr style={styles.rowBold}><td>Gross revenue</td><td style={styles.num}>{fmt(metrics.grossRevenue)}</td></tr>
                  <tr><td>Processing fees ({pct(channel.cut)})</td><td style={styles.num}>-{fmt(metrics.processingFees)}</td></tr>
                  <tr style={styles.rowBold}><td>Net revenue</td><td style={styles.num}>{fmt(metrics.netRevenue)}</td></tr>
                  <tr><td colSpan={2} style={{ height: '8px' }} /></tr>
                  {metrics.costBreakdown.filter(c => c.amount > 0).map(c => (
                    <tr key={c.key}><td>{c.label}</td><td style={styles.num}>-{fmt(c.amount)}</td></tr>
                  ))}
                  <tr style={styles.rowBold}><td>Total costs</td><td style={styles.num}>-{fmt(metrics.totalCosts)}</td></tr>
                  <tr><td colSpan={2} style={{ height: '8px' }} /></tr>
                  <tr style={{ ...styles.rowBold, color: metrics.margin >= 0 ? '#4ade80' : '#f87171' }}>
                    <td>Margin</td>
                    <td style={styles.num}>{fmt(metrics.margin)} ({pct(metrics.marginPct)})</td>
                  </tr>
                </tbody>
              </table>

              <h3 style={{ marginTop: '2rem' }}>Waterfall</h3>
              <div style={styles.waterfallContainer}>
                {[
                  { label: 'Gross', value: metrics.grossRevenue, color: '#646cff' },
                  { label: 'Fees', value: -metrics.processingFees, color: '#f87171' },
                  { label: 'Net', value: metrics.netRevenue, color: '#818cf8' },
                  { label: 'Costs', value: -metrics.totalCosts, color: '#f87171' },
                  { label: 'Margin', value: metrics.margin, color: metrics.margin >= 0 ? '#4ade80' : '#f87171' },
                ].map(bar => (
                  <div key={bar.label} style={styles.waterfallBar}>
                    <div style={styles.waterfallLabel}>{bar.label}</div>
                    <div style={{
                      height: `${Math.abs(bar.value) / maxWaterfall * 120}px`,
                      backgroundColor: bar.color,
                      borderRadius: '4px 4px 0 0',
                      minHeight: '2px',
                      width: '40px',
                    }} />
                    <div style={styles.waterfallValue}>{fmt(bar.value)}</div>
                  </div>
                ))}
              </div>

              <h3 style={{ marginTop: '2rem' }}>12-Month Fleet Projection</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Mo</th>
                      <th style={styles.th}>Gyms</th>
                      <th style={styles.th}>Revenue</th>
                      <th style={styles.th}>Costs</th>
                      <th style={styles.th}>Margin</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.fleet.map(f => (
                      <tr key={f.month}>
                        <td style={styles.num}>{f.month}</td>
                        <td style={styles.num}>{f.gyms}</td>
                        <td style={styles.num}>{fmt(f.revenue)}</td>
                        <td style={styles.num}>{fmt(f.costs)}</td>
                        <td style={{ ...styles.num, color: f.margin >= 0 ? '#4ade80' : '#f87171' }}>
                          {fmt(f.margin)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Unit Economics Tab */}
      {tab === 'unit' && (
        <div style={styles.content}>
          <h3>Per-Unit Economics</h3>
          <table style={styles.table}>
            <tbody>
              <tr><td>Total gyms</td><td style={styles.num}>{gymCount}</td></tr>
              <tr><td>Total athletes</td><td style={styles.num}>{metrics.totalAthletes.toLocaleString()}</td></tr>
              <tr><td>Total clips / month</td><td style={styles.num}>{metrics.totalClips.toLocaleString()}</td></tr>
              <tr><td colSpan={2} style={{ height: '12px' }} /></tr>
              <tr style={styles.rowBold}><td>Revenue / gym</td><td style={styles.num}>{fmt(metrics.revenuePerGym)}</td></tr>
              <tr style={styles.rowBold}><td>Cost / gym</td><td style={styles.num}>{fmt(metrics.costPerGym)}</td></tr>
              <tr style={{ ...styles.rowBold, color: metrics.marginPerGym >= 0 ? '#4ade80' : '#f87171' }}>
                <td>Margin / gym</td><td style={styles.num}>{fmt(metrics.marginPerGym)}</td>
              </tr>
              <tr><td colSpan={2} style={{ height: '12px' }} /></tr>
              <tr><td>Revenue / athlete</td><td style={styles.num}>{fmt(metrics.revenuePerAthlete)}</td></tr>
            </tbody>
          </table>

          <h3 style={{ marginTop: '2rem' }}>Waterfall: Per-Gym Breakdown</h3>
          <div style={styles.waterfallContainer}>
            {[
              { label: 'Revenue', value: metrics.revenuePerGym, color: '#646cff' },
              { label: 'Costs', value: -metrics.costPerGym, color: '#f87171' },
              { label: 'Margin', value: metrics.marginPerGym, color: metrics.marginPerGym >= 0 ? '#4ade80' : '#f87171' },
            ].map(bar => {
              const maxVal = Math.max(metrics.revenuePerGym, 1)
              return (
                <div key={bar.label} style={styles.waterfallBar}>
                  <div style={styles.waterfallLabel}>{bar.label}</div>
                  <div style={{
                    height: `${Math.abs(bar.value) / maxVal * 120}px`,
                    backgroundColor: bar.color,
                    borderRadius: '4px 4px 0 0',
                    minHeight: '2px',
                    width: '40px',
                  }} />
                  <div style={styles.waterfallValue}>{fmt(bar.value)}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Sensitivity Tab */}
      {tab === 'sensitivity' && (
        <div style={styles.content}>
          <h3>Margin Sensitivity: Gyms x Athletes/Gym</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Gyms \\ Athletes</th>
                  {[30, 50, 80, 120, 200].map(a => (
                    <th key={a} style={styles.th}>{a}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sensitivityData.map((row, ri) => (
                  <tr key={ri}>
                    <td style={{ ...styles.num, fontWeight: 'bold' }}>{row[0].gyms}</td>
                    {row.map((cell, ci) => (
                      <td key={ci} style={{
                        ...styles.num,
                        backgroundColor: cell.margin >= 0
                          ? `rgba(74, 222, 128, ${Math.min(cell.marginPct, 0.8) * 0.5})`
                          : `rgba(248, 113, 113, ${Math.min(Math.abs(cell.marginPct), 0.8) * 0.5})`,
                      }}>
                        {fmt(cell.margin)}
                        <br />
                        <span style={{ fontSize: '0.75em', opacity: 0.7 }}>{pct(cell.marginPct)}</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Notes Tab */}
      {tab === 'notes' && (
        <div style={styles.content}>
          <h3>Assumptions & Notes</h3>
          <textarea
            value={notes}
            onChange={e => setNotes(e.target.value)}
            placeholder="Document your assumptions, scenarios, and notes here..."
            style={styles.textarea}
          />
        </div>
      )}
    </div>
  )
}

const styles = {
  root: {
    padding: '1rem 2rem',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  header: {
    marginTop: 0,
    marginBottom: '0.5rem',
    fontSize: '1.25rem',
    fontWeight: 600,
    opacity: 0.8,
  },
  tabs: {
    display: 'flex',
    gap: '4px',
    marginBottom: '1.5rem',
    borderBottom: '1px solid #333',
    paddingBottom: '8px',
  },
  tab: {
    padding: '0.4em 1em',
    borderRadius: '6px 6px 0 0',
    border: '1px solid transparent',
    backgroundColor: 'transparent',
    color: 'inherit',
    cursor: 'pointer',
    fontSize: '0.9em',
    fontFamily: 'inherit',
  },
  tabActive: {
    backgroundColor: '#333',
    borderColor: '#555',
  },
  content: {
    minHeight: '400px',
  },
  row: {
    display: 'flex',
    gap: '2rem',
    flexWrap: 'wrap',
  },
  col: {
    flex: '1 1 400px',
    minWidth: '300px',
  },
  sliderLabel: {
    display: 'block',
    marginBottom: '0.5rem',
    fontSize: '0.9em',
  },
  slider: {
    display: 'block',
    width: '100%',
    marginTop: '2px',
  },
  radioLabel: {
    display: 'block',
    marginBottom: '0.25rem',
    fontSize: '0.9em',
  },
  checkLabel: {
    display: 'block',
    marginBottom: '0.4rem',
    fontSize: '0.9em',
  },
  table: {
    borderCollapse: 'collapse',
    width: '100%',
    fontSize: '0.9em',
  },
  th: {
    textAlign: 'right',
    padding: '4px 8px',
    borderBottom: '1px solid #444',
    fontSize: '0.85em',
    opacity: 0.7,
  },
  num: {
    textAlign: 'right',
    padding: '3px 8px',
    fontVariantNumeric: 'tabular-nums',
  },
  rowBold: {
    fontWeight: 'bold',
  },
  waterfallContainer: {
    display: 'flex',
    alignItems: 'flex-end',
    gap: '1rem',
    padding: '1rem 0',
  },
  waterfallBar: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
  },
  waterfallLabel: {
    fontSize: '0.75em',
    opacity: 0.7,
  },
  waterfallValue: {
    fontSize: '0.8em',
    fontWeight: 'bold',
  },
  textarea: {
    width: '100%',
    minHeight: '300px',
    padding: '1rem',
    borderRadius: '8px',
    border: '1px solid #444',
    backgroundColor: '#1a1a1a',
    color: 'inherit',
    fontFamily: 'inherit',
    fontSize: '0.95em',
    resize: 'vertical',
    boxSizing: 'border-box',
  },
}
