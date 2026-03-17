import { useState, useMemo } from 'react'

// ── Constants ──────────────────────────────────────────────────────────────

const BILLING_CHANNELS = [
  { key: 'web',          label: 'Web (Stripe only)',         platformCut: 0 },
  { key: 'apple_small',  label: 'Apple small biz (≤$1M)',    platformCut: 0.15 },
  { key: 'apple_std',    label: 'Apple standard',            platformCut: 0.30 },
  { key: 'google_small', label: 'Google Play small biz',     platformCut: 0.15 },
]

const SENSITIVITY_PRICES = [3, 5, 8, 10, 12, 15, 20, 25]

const NOTES_CARDS = [
  { title: 'Gear markup', body: 'Replacement gear modeled at 2.2\u00d7 COGS for rashguards, 2.5\u00d7 for patches. Adjust COGS sliders once you have a manufacturing quote.' },
  { title: 'Coach marketplace', body: 'Revenue = athletes \u00d7 months \u00d7 reviews/athlete/year \u00d7 review price \u00d7 your cut %. Treat as Year 2+ upside, not baseline.' },
  { title: 'CAC', body: 'Covers everything to close a gym: demos, travel, onboarding labor, your time. $500 is conservative for founder-led sales. Budget $1,000\u2013$2,500 once you hire sales.' },
  { title: 'Headcount', body: 'Model triggers one hire at your chosen gym count, shared across all gyms. One ops/support person at $60K can plausibly support 20\u201330 gyms.' },
  { title: 'Break-even price', body: 'Floor price = total 3-yr costs \u00f7 total athlete-months \u00d7 net-per-dollar factor. Price above this for sustainable margin.' },
  { title: 'LTV', body: 'Net profit \u00f7 Year 1 athlete count over 3 years. Healthy SaaS LTV:CAC ratio is 3:1 or better. Check the Unit Economics tab.' },
  { title: 'Web billing', body: 'Highly recommended. Stripe-only billing via your website avoids 15\u201330% platform cuts. Users sign up on web, log in on app. Spotify and Netflix use this model on iOS.' },
]

// ── Formatting helpers ─────────────────────────────────────────────────────

function fmt(n) {
  const abs = Math.abs(n)
  const sign = n < 0 ? '-' : ''
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`
  return `${sign}$${Math.round(abs)}`
}

// ── Calculation engine ─────────────────────────────────────────────────────

function calcModel(p, overrideAthletePrice) {
  const athletePrice = overrideAthletePrice ?? p.athletePrice
  const ret = 1 - p.annualChurn / 100
  const sr = p.athleteSignupRate / 100
  const m = [p.members, Math.round(p.members * ret), Math.round(p.members * ret * ret)]
  const a = m.map(mi => Math.round(mi * sr))
  const totalAthleteMonths = (a[0] + a[1] + a[2]) * 12

  // Gym sub
  const gymSubRev = p.gymSub ? p.gymPrice * 12 * 3 : 0

  // Athlete sub
  const athleteSubGross = p.athleteSub ? totalAthleteMonths * athletePrice : 0
  const billing = BILLING_CHANNELS.find(c => c.key === p.billingChannel)
  const appFees = athleteSubGross * billing.platformCut
  const stripeFees = (athleteSubGross - appFees) * 0.029 + 0.30 * totalAthleteMonths
  const athleteSubNet = athleteSubGross - appFees - stripeFees

  // Gear replacement
  const rashRetail = p.rashCogs * 2.2
  const patchRetail = p.patchCogs * 2.5
  const gearReplRev = p.gearRepl
    ? Math.round(a[1] * 0.40 * rashRetail) + Math.round(a[1] * 0.10 * patchRetail)
    + Math.round(a[2] * 0.50 * rashRetail) + Math.round(a[2] * 0.30 * patchRetail)
    : 0
  const gearReplCogs = p.gearRepl
    ? Math.round(a[1] * 0.40 * p.rashCogs) + Math.round(a[1] * 0.10 * p.patchCogs)
    + Math.round(a[2] * 0.50 * p.rashCogs) + Math.round(a[2] * 0.30 * p.patchCogs)
    : 0

  // Starter pack
  const starterCogsTotal = p.freeStarterPack ? a[0] * p.starterCogs : 0

  // Coach marketplace
  const coachRev = p.coachMarket
    ? (a[0] + a[1] + a[2]) * 12 * (p.coachTxnPerYear / 12) * p.coachTxnValue * (p.coachCut / 100)
    : 0

  // Costs
  const hwAmort = (p.hwCost / p.hwMonths) * 12 * 3
  const opsCost = p.opsMo * 12 * 3
  const cacCost = p.includeCAC ? p.cacPerGym : 0
  const headcostPerGym = (p.includeHeadcount && p.gymCount >= p.headcountAt)
    ? (p.headcountSalary / p.gymCount) * 3 : 0

  const totalCost = hwAmort + opsCost + cacCost + starterCogsTotal + gearReplCogs + headcostPerGym
  const totalRevenue = gymSubRev + athleteSubNet + gearReplRev + coachRev
  const grossRevenue = gymSubRev + athleteSubGross + gearReplRev + coachRev
  const profit = totalRevenue - totalCost
  const margin = grossRevenue > 0 ? (profit / grossRevenue) * 100 : 0

  // HW payback
  const monthlyNetSub = (gymSubRev + athleteSubNet) / 36
  const hwPayback = monthlyNetSub > 0 ? Math.round((p.hwCost + cacCost) / monthlyNetSub) : null

  // Break-even
  const netPerDollar = totalAthleteMonths > 0
    ? (1 - billing.platformCut) * (1 - 0.029) : 0
  const bep = netPerDollar > 0 && totalAthleteMonths > 0
    ? totalCost / (totalAthleteMonths * netPerDollar) : null

  // LTV
  const ltv = a[0] > 0 ? profit / a[0] : 0

  // Fleet
  const fleetRevenue = grossRevenue * p.gymCount
  const fleetProfit = profit * p.gymCount

  // Gear vs HW coverage
  const totalHwFleet = p.hwCost * p.gymCount
  const totalGearRevFleet = gearReplRev * p.gymCount
  const gearHwCoverage = totalHwFleet > 0 ? (totalGearRevFleet / totalHwFleet) * 100 : 0

  // Headcount
  const staffNeeded = (p.includeHeadcount && p.gymCount >= p.headcountAt) ? 1 : 0
  const staffCost3yr = staffNeeded * p.headcountSalary * 3

  return {
    m, a, totalAthleteMonths,
    gymSubRev, athleteSubGross, appFees, stripeFees, athleteSubNet,
    gearReplRev, gearReplCogs, starterCogsTotal, coachRev,
    hwAmort, opsCost, cacCost, headcostPerGym,
    totalCost, totalRevenue, grossRevenue, profit, margin,
    hwPayback, bep, ltv,
    fleetRevenue, fleetProfit, gearHwCoverage,
    staffNeeded, staffCost3yr,
    billing,
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export default function PricingDashboard() {
  const [tab, setTab] = useState('model')

  // Revenue toggles
  const [gymSub, setGymSub] = useState(false)
  const [athleteSub, setAthleteSub] = useState(true)
  const [gearRepl, setGearRepl] = useState(true)
  const [coachMarket, setCoachMarket] = useState(false)

  // Cost toggles
  const [includeCAC, setIncludeCAC] = useState(true)
  const [includeHeadcount, setIncludeHeadcount] = useState(false)
  const [freeStarterPack, setFreeStarterPack] = useState(true)

  // Gym sub
  const [gymPrice, setGymPrice] = useState(299)

  // Athlete sub
  const [athletePrice, setAthletePrice] = useState(10)
  const [billingChannel, setBillingChannel] = useState('web')
  const [athleteSignupRate, setAthleteSignupRate] = useState(70)

  // Coach marketplace
  const [coachTxnPerYear, setCoachTxnPerYear] = useState(2)
  const [coachTxnValue, setCoachTxnValue] = useState(40)
  const [coachCut, setCoachCut] = useState(20)

  // Gym profile
  const [members, setMembers] = useState(120)
  const [annualChurn, setAnnualChurn] = useState(20)
  const [gymCount, setGymCount] = useState(10)

  // Ops + costs
  const [hwCost, setHwCost] = useState(1800)
  const [hwMonths, setHwMonths] = useState(36)
  const [opsMo, setOpsMo] = useState(40)
  const [cacPerGym, setCacPerGym] = useState(500)
  const [headcountAt, setHeadcountAt] = useState(20)
  const [headcountSalary, setHeadcountSalary] = useState(60000)
  const [starterCogs, setStarterCogs] = useState(25)
  const [rashCogs, setRashCogs] = useState(20)
  const [patchCogs, setPatchCogs] = useState(5)

  const deps = [
    gymSub, athleteSub, gearRepl, coachMarket,
    includeCAC, includeHeadcount, freeStarterPack,
    gymPrice, athletePrice, billingChannel, athleteSignupRate,
    coachTxnPerYear, coachTxnValue, coachCut,
    members, annualChurn, gymCount,
    hwCost, hwMonths, opsMo, cacPerGym,
    headcountAt, headcountSalary, starterCogs, rashCogs, patchCogs,
  ]

  const params = useMemo(() => ({
    gymSub, athleteSub, gearRepl, coachMarket,
    includeCAC, includeHeadcount, freeStarterPack,
    gymPrice, athletePrice, billingChannel, athleteSignupRate,
    coachTxnPerYear, coachTxnValue, coachCut,
    members, annualChurn, gymCount,
    hwCost, hwMonths, opsMo, cacPerGym,
    headcountAt, headcountSalary, starterCogs, rashCogs, patchCogs,
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), deps)

  const M = useMemo(() => calcModel(params), [params])

  // ── Waterfall bars ─────────────────────────────────────────────────────

  const waterfallBars = useMemo(() => {
    const bars = []
    if (gymSub) bars.push({ label: 'Gym subscriptions', value: M.gymSubRev, color: '#4f8ff7' })
    if (athleteSub) {
      bars.push({ label: 'Athlete subs gross', value: M.athleteSubGross, color: '#4ade80' })
      if (M.billing.platformCut > 0) bars.push({ label: 'Platform cut', value: -M.appFees, color: '#f87171' })
      bars.push({ label: 'Stripe fees', value: -M.stripeFees, color: '#fb923c' })
    }
    if (gearRepl) {
      bars.push({ label: 'Gear replacements', value: M.gearReplRev, color: '#fbbf24' })
      bars.push({ label: 'Gear COGS', value: -M.gearReplCogs, color: '#f97316' })
    }
    if (freeStarterPack) bars.push({ label: 'Starter pack COGS', value: -M.starterCogsTotal, color: '#b91c1c' })
    if (coachMarket) bars.push({ label: 'Coach marketplace', value: M.coachRev, color: '#a78bfa' })
    bars.push({ label: 'Hardware + ops', value: -(M.hwAmort + M.opsCost), color: '#6b7280' })
    if (includeCAC) bars.push({ label: 'CAC', value: -M.cacCost, color: '#be185d' })
    if (includeHeadcount && gymCount >= headcountAt) {
      bars.push({ label: 'Headcount share', value: -M.headcostPerGym, color: '#7c3aed' })
    }
    return bars
  }, [M, gymSub, athleteSub, gearRepl, freeStarterPack, coachMarket, includeCAC, includeHeadcount, gymCount, headcountAt])

  const maxBar = Math.max(...waterfallBars.map(b => Math.abs(b.value)), 1)

  // ── Sensitivity rows ───────────────────────────────────────────────────

  const sensitivityRows = useMemo(() => {
    if (!athleteSub) return null
    return SENSITIVITY_PRICES.map(price => {
      const r = calcModel(params, price)
      return { price, profit: r.profit, margin: r.margin }
    })
  }, [params, athleteSub])

  // ── Render helpers ─────────────────────────────────────────────────────

  const Toggle = ({ checked, onChange, label, hint }) => (
    <label style={s.toggleRow}>
      <div
        style={{
          ...s.toggleTrack,
          backgroundColor: checked ? '#4f8ff7' : '#444',
        }}
        onClick={() => onChange(!checked)}
      >
        <div style={{
          ...s.toggleThumb,
          transform: checked ? 'translateX(18px)' : 'translateX(2px)',
        }} />
      </div>
      <div>
        <div style={s.toggleLabel}>{label}</div>
        {hint && <div style={s.hint}>{hint}</div>}
      </div>
    </label>
  )

  const Slider = ({ label, value, min, max, step, onChange, hint, prefix, suffix }) => (
    <div style={s.sliderBlock}>
      <div style={s.sliderHeader}>
        <span>{label}</span>
        <strong>{prefix || ''}{typeof value === 'number' ? value.toLocaleString() : value}{suffix || ''}</strong>
      </div>
      <input type="range" min={min} max={max} step={step || 1} value={value}
        onChange={e => onChange(+e.target.value)} style={s.slider} />
      {hint && <div style={s.hint}>{hint}</div>}
    </div>
  )

  const KpiCard = ({ label, value, warn }) => (
    <div style={{ ...s.kpiCard, borderColor: warn ? '#f87171' : '#333' }}>
      <div style={s.kpiValue}>{value}</div>
      <div style={s.kpiLabel}>{label}</div>
    </div>
  )

  const TableRow = ({ label, value, off }) => (
    <div style={s.tableRow}>
      <span>{label}</span>
      <span style={{ fontVariantNumeric: 'tabular-nums', opacity: off ? 0.4 : 1 }}>
        {off ? 'off' : value}
      </span>
    </div>
  )

  // ── Tab: Model ─────────────────────────────────────────────────────────

  const renderModel = () => (
    <div style={s.twoCol}>
      {/* Left column */}
      <div style={s.leftCol}>
        <h3 style={s.sectionTitle}>Revenue streams</h3>
        <Toggle checked={gymSub} onChange={setGymSub}
          label="Gym pays subscription" hint="Charge the gym owner a flat monthly fee" />
        <Toggle checked={athleteSub} onChange={setAthleteSub}
          label="Athlete pays subscription" hint="Athletes pay monthly via app or web" />
        <Toggle checked={gearRepl} onChange={setGearRepl}
          label="Gear replacement revenue" hint="Sell replacement rashguards + patches in Y2, Y3" />
        <Toggle checked={coachMarket} onChange={setCoachMarket}
          label="Coach review marketplace" hint="Take a % of coach-athlete clip review transactions" />

        <h3 style={s.sectionTitle}>Costs</h3>
        <Toggle checked={includeCAC} onChange={setIncludeCAC}
          label="Include CAC" hint="Customer acquisition cost per gym" />
        <Toggle checked={includeHeadcount} onChange={setIncludeHeadcount}
          label="Include headcount" hint="First hire at a gym count threshold" />
        <Toggle checked={freeStarterPack} onChange={setFreeStarterPack}
          label="Free starter pack at signup" hint="Absorb COGS \u2014 rashguard + patch kit" />

        {/* Gym sub section */}
        {gymSub && (
          <div style={s.conditionalSection}>
            <h4 style={s.subSectionTitle}>Gym subscription</h4>
            <Slider label="Gym price / month" value={gymPrice} min={49} max={999} step={10}
              onChange={setGymPrice} prefix="$" />
          </div>
        )}

        {/* Athlete sub section */}
        {athleteSub && (
          <div style={s.conditionalSection}>
            <h4 style={s.subSectionTitle}>Athlete subscription</h4>
            <Slider label="Athlete price / month" value={athletePrice} min={2} max={30} step={1}
              onChange={setAthletePrice} prefix="$" hint="Gym membership ~$165/mo" />
            <div style={s.hint}>Billing channel</div>
            <div style={s.channelRow}>
              {BILLING_CHANNELS.map(ch => (
                <button key={ch.key}
                  style={{
                    ...s.channelBtn,
                    ...(billingChannel === ch.key ? s.channelBtnActive : {}),
                  }}
                  onClick={() => setBillingChannel(ch.key)}
                >
                  <div style={{ fontSize: '0.8em', fontWeight: 600 }}>{ch.label}</div>
                  <div style={{ fontSize: '0.7em', opacity: 0.7 }}>
                    {ch.platformCut === 0 ? '0% platform cut' : `${ch.platformCut * 100}% platform cut`}
                  </div>
                </button>
              ))}
            </div>
            <Slider label="Athlete signup rate" value={athleteSignupRate} min={20} max={100} step={5}
              onChange={setAthleteSignupRate} suffix="%" hint="% of gym members who subscribe" />
          </div>
        )}

        {/* Coach marketplace section */}
        {coachMarket && (
          <div style={s.conditionalSection}>
            <h4 style={s.subSectionTitle}>Coach marketplace</h4>
            <Slider label="Reviews / athlete / year" value={coachTxnPerYear} min={1} max={12}
              onChange={setCoachTxnPerYear} />
            <Slider label="Avg review price" value={coachTxnValue} min={10} max={150} step={5}
              onChange={setCoachTxnValue} prefix="$" />
            <Slider label="Your cut" value={coachCut} min={10} max={40}
              onChange={setCoachCut} suffix="%" />
          </div>
        )}

        {/* Gym profile — always visible */}
        <h3 style={s.sectionTitle}>Gym profile</h3>
        <Slider label="Active members at gym" value={members} min={40} max={300} step={10}
          onChange={setMembers} />
        <Slider label="Annual member churn" value={annualChurn} min={5} max={50}
          onChange={setAnnualChurn} suffix="%" hint="Industry avg: 20\u201333%" />
        <Slider label="Gyms onboarded (yr 3)" value={gymCount} min={1} max={100}
          onChange={setGymCount} />

        {/* Ops + costs — always visible */}
        <h3 style={s.sectionTitle}>Ops + costs</h3>
        <Slider label="Hardware / gym" value={hwCost} min={500} max={5000} step={100}
          onChange={setHwCost} prefix="$" />
        <Slider label="Amortization months" value={hwMonths} min={12} max={60}
          onChange={setHwMonths} />
        <Slider label="Ops / gym / month" value={opsMo} min={10} max={200} step={5}
          onChange={setOpsMo} prefix="$" />
        {includeCAC && (
          <Slider label="CAC per gym" value={cacPerGym} min={0} max={5000} step={50}
            onChange={setCacPerGym} prefix="$" />
        )}
        {includeHeadcount && (
          <>
            <Slider label="Hire at gym count" value={headcountAt} min={5} max={50}
              onChange={setHeadcountAt} />
            <Slider label="Annual salary" value={headcountSalary} min={40000} max={120000} step={5000}
              onChange={setHeadcountSalary} prefix="$" />
          </>
        )}
        {(freeStarterPack || gearRepl) && (
          <Slider label="Starter pack COGS" value={starterCogs} min={10} max={60}
            onChange={setStarterCogs} prefix="$" />
        )}
        {gearRepl && (
          <>
            <Slider label="Rashguard COGS" value={rashCogs} min={10} max={50}
              onChange={setRashCogs} prefix="$" />
            <Slider label="Patch COGS" value={patchCogs} min={2} max={20}
              onChange={setPatchCogs} prefix="$" />
          </>
        )}
      </div>

      {/* Right column */}
      <div style={s.rightCol}>
        {/* KPI cards */}
        <div style={s.kpiGrid}>
          <KpiCard label="3-yr revenue / gym (gross)" value={fmt(M.grossRevenue)} />
          <KpiCard label="3-yr net profit / gym"
            value={`${fmt(M.profit)} (${Math.round(M.margin)}%)`} />
          <KpiCard label="HW payback (months)"
            value={M.hwPayback !== null ? `${M.hwPayback} mo` : 'N/A'} />
          <KpiCard label="Fleet profit (net, 3yr)"
            value={fmt(M.fleetProfit)} warn={M.fleetProfit < 0} />
        </div>

        {/* Waterfall */}
        <h3 style={s.sectionTitle}>Revenue vs cost waterfall</h3>
        <div style={s.waterfallList}>
          {waterfallBars.map(bar => (
            <div key={bar.label} style={s.waterfallRow}>
              <div style={s.waterfallLabel}>{bar.label}</div>
              <div style={s.waterfallBarTrack}>
                <div style={{
                  width: `${(Math.abs(bar.value) / maxBar) * 100}%`,
                  backgroundColor: bar.color,
                  height: '18px',
                  borderRadius: '3px',
                  minWidth: bar.value !== 0 ? '2px' : '0',
                }} />
              </div>
              <div style={s.waterfallValue}>{fmt(bar.value)}</div>
            </div>
          ))}
          <div style={{ borderTop: '1px solid #444', margin: '6px 0' }} />
          <div style={s.waterfallRow}>
            <div style={{ ...s.waterfallLabel, fontWeight: 700 }}>Net profit</div>
            <div style={s.waterfallBarTrack}>
              <div style={{
                width: `${(Math.abs(M.profit) / maxBar) * 100}%`,
                backgroundColor: M.profit >= 0 ? '#4ade80' : '#f87171',
                height: '18px',
                borderRadius: '3px',
                minWidth: '2px',
              }} />
            </div>
            <div style={{ ...s.waterfallValue, fontWeight: 700, color: M.profit >= 0 ? '#4ade80' : '#f87171' }}>
              {fmt(M.profit)}
            </div>
          </div>
        </div>

        {/* Fleet section */}
        <h3 style={s.sectionTitle}>Fleet &middot; {gymCount} gyms &middot; 3 years</h3>
        <div style={s.kpiGrid}>
          <KpiCard label="Total fleet gross revenue" value={fmt(M.fleetRevenue)} />
          <KpiCard label="Total fleet net profit" value={fmt(M.fleetProfit)} warn={M.fleetProfit < 0} />
        </div>
        {gearRepl && (
          <p style={s.fleetNote}>
            Gear revenue alone covers ~{Math.round(M.gearHwCoverage)}% of total hardware costs across the fleet.
          </p>
        )}
      </div>
    </div>
  )

  // ── Tab: Unit Economics ────────────────────────────────────────────────

  const renderUnit = () => (
    <div style={s.twoCol}>
      <div style={s.leftCol}>
        <h3 style={s.sectionTitle}>Per-gym economics (3 yr)</h3>
        <TableRow label="Gross revenue" value={fmt(M.grossRevenue)} />
        <TableRow label="Platform + Stripe fees" value={fmt(-(M.appFees + M.stripeFees))}
          off={!athleteSub} />
        <TableRow label="Gear COGS" value={fmt(-M.gearReplCogs)} off={!gearRepl} />
        <TableRow label="Hardware amortized" value={fmt(-M.hwAmort)} />
        <TableRow label="Ops (3 yr)" value={fmt(-M.opsCost)} />
        <TableRow label="CAC" value={fmt(-M.cacCost)} off={!includeCAC} />
        <TableRow label="Headcount share" value={fmt(-M.headcostPerGym)} off={!includeHeadcount} />
        <TableRow label="Starter pack COGS" value={fmt(-M.starterCogsTotal)} off={!freeStarterPack} />
        <div style={{ borderTop: '1px solid #444', margin: '8px 0' }} />
        <TableRow label="Net profit" value={fmt(M.profit)} />
        <TableRow label="Margin %" value={`${Math.round(M.margin)}%`} />
      </div>

      <div style={s.rightCol}>
        <h3 style={s.sectionTitle}>Fleet economics ({gymCount} gyms, 3 yr)</h3>
        <TableRow label="Total fleet revenue" value={fmt(M.fleetRevenue)} />
        <TableRow label="Total fleet profit" value={fmt(M.fleetProfit)} />
        <TableRow label="Blended margin %" value={`${Math.round(M.margin)}%`} />
        <TableRow label="Total hardware cost" value={fmt(hwCost * gymCount)} />
        <TableRow label="Total CAC" value={fmt(M.cacCost * gymCount)} off={!includeCAC} />
        <TableRow label="Staff needed (FTE)" value={M.staffNeeded} off={!includeHeadcount} />
        <TableRow label="Staff cost (3 yr)" value={fmt(M.staffCost3yr)} off={!includeHeadcount} />
      </div>
    </div>
  )

  // ── Tab: Sensitivity ───────────────────────────────────────────────────

  const renderSensitivity = () => {
    if (!athleteSub) {
      return (
        <div style={{ padding: '2rem', textAlign: 'center', opacity: 0.6 }}>
          Enable athlete subscription to see price sensitivity.
        </div>
      )
    }

    const maxProfit = 50000

    return (
      <div>
        <h3 style={s.sectionTitle}>3-year net profit per gym across athlete price points</h3>
        <div style={s.sensitivityList}>
          {sensitivityRows.map(row => (
            <div key={row.price} style={s.sensitivityRow}
              onClick={() => { setAthletePrice(row.price); setTab('model') }}>
              <div style={s.sensitivityPrice}>${row.price}/mo</div>
              <div style={s.sensitivityBarTrack}>
                <div style={{
                  width: `${Math.min(Math.abs(row.profit) / maxProfit * 100, 100)}%`,
                  backgroundColor: row.profit >= 0 ? '#4ade80' : '#f87171',
                  height: '22px',
                  borderRadius: '3px',
                  minWidth: '2px',
                }} />
              </div>
              <div style={{
                ...s.sensitivityValue,
                color: row.profit >= 0 ? '#4ade80' : '#f87171',
              }}>
                {fmt(row.profit)}
              </div>
              <div style={s.sensitivityMargin}>{Math.round(row.margin)}%</div>
            </div>
          ))}
        </div>
        <p style={s.hint}>Tap any row to set that price and return to the Model tab.</p>
      </div>
    )
  }

  // ── Tab: Notes ─────────────────────────────────────────────────────────

  const renderNotes = () => (
    <div style={s.notesGrid}>
      {NOTES_CARDS.map(card => (
        <div key={card.title} style={s.noteCard}>
          <div style={s.noteTitle}>{card.title}</div>
          <div style={s.noteBody}>{card.body}</div>
        </div>
      ))}
    </div>
  )

  // ── Main render ────────────────────────────────────────────────────────

  return (
    <div style={s.root}>
      <h2 style={s.header}>Roll Tracker &middot; Admin &mdash; Business model simulator</h2>
      <div style={s.subtitle}>3-year projection &middot; single gym unless noted</div>

      <div style={s.tabs}>
        {[
          { key: 'model', label: 'Model' },
          { key: 'unit', label: 'Unit Economics' },
          { key: 'sensitivity', label: 'Sensitivity' },
          { key: 'notes', label: 'Notes' },
        ].map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            style={{ ...s.tab, ...(tab === t.key ? s.tabActive : {}) }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'model' && renderModel()}
      {tab === 'unit' && renderUnit()}
      {tab === 'sensitivity' && renderSensitivity()}
      {tab === 'notes' && renderNotes()}
    </div>
  )
}

// ── Styles ──────────────────────────────────────────────────────────────────

const s = {
  root: { padding: '1rem 2rem', maxWidth: '1200px', margin: '0 auto' },
  header: { marginTop: 0, marginBottom: '2px', fontSize: '1.25rem', fontWeight: 600, opacity: 0.85 },
  subtitle: { fontSize: '0.85rem', opacity: 0.5, marginBottom: '1rem' },

  tabs: {
    display: 'flex', gap: '4px', marginBottom: '1.5rem',
    borderBottom: '1px solid #333', paddingBottom: '8px',
  },
  tab: {
    padding: '0.4em 1em', borderRadius: '6px 6px 0 0',
    border: '1px solid transparent', backgroundColor: 'transparent',
    color: 'inherit', cursor: 'pointer', fontSize: '0.9em', fontFamily: 'inherit',
  },
  tabActive: { backgroundColor: '#333', borderColor: '#555' },

  twoCol: { display: 'flex', gap: '2rem', flexWrap: 'wrap' },
  leftCol: { flex: '1 1 380px', minWidth: '300px' },
  rightCol: { flex: '1 1 480px', minWidth: '340px' },

  sectionTitle: { fontSize: '1rem', fontWeight: 600, marginTop: '1.5rem', marginBottom: '0.75rem' },
  subSectionTitle: { fontSize: '0.9rem', fontWeight: 600, marginTop: '0.5rem', marginBottom: '0.5rem', opacity: 0.8 },
  conditionalSection: { paddingLeft: '12px', borderLeft: '2px solid #333', marginBottom: '0.75rem' },

  // Toggle
  toggleRow: {
    display: 'flex', alignItems: 'flex-start', gap: '10px',
    marginBottom: '0.75rem', cursor: 'pointer',
  },
  toggleTrack: {
    width: '38px', height: '20px', borderRadius: '10px',
    position: 'relative', flexShrink: 0, cursor: 'pointer',
    transition: 'background-color 0.15s',
  },
  toggleThumb: {
    width: '16px', height: '16px', borderRadius: '50%',
    backgroundColor: '#fff', position: 'absolute', top: '2px',
    transition: 'transform 0.15s',
  },
  toggleLabel: { fontSize: '0.9em', fontWeight: 500 },
  hint: { fontSize: '0.75em', opacity: 0.5, marginTop: '2px' },

  // Slider
  sliderBlock: { marginBottom: '0.75rem' },
  sliderHeader: {
    display: 'flex', justifyContent: 'space-between',
    fontSize: '0.85em', marginBottom: '2px',
  },
  slider: { display: 'block', width: '100%', marginTop: '2px' },

  // Billing channel
  channelRow: { display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '0.75rem' },
  channelBtn: {
    flex: '1 1 0', padding: '8px 6px', borderRadius: '6px',
    border: '1px solid #444', backgroundColor: '#1a1a1a',
    color: 'inherit', cursor: 'pointer', textAlign: 'center',
    fontFamily: 'inherit', minWidth: '110px',
  },
  channelBtnActive: {
    borderColor: '#4f8ff7', backgroundColor: '#1e3a5f',
  },

  // KPI cards
  kpiGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
    gap: '10px', marginBottom: '1rem',
  },
  kpiCard: {
    padding: '12px', borderRadius: '8px',
    border: '1px solid #333', backgroundColor: '#1a1a1a',
  },
  kpiValue: { fontSize: '1.2em', fontWeight: 700, fontVariantNumeric: 'tabular-nums' },
  kpiLabel: { fontSize: '0.75em', opacity: 0.6, marginTop: '4px' },

  // Waterfall (horizontal)
  waterfallList: { display: 'flex', flexDirection: 'column', gap: '4px' },
  waterfallRow: { display: 'flex', alignItems: 'center', gap: '8px' },
  waterfallLabel: { width: '160px', fontSize: '0.8em', flexShrink: 0 },
  waterfallBarTrack: { flex: 1, height: '18px', borderRadius: '3px' },
  waterfallValue: {
    width: '70px', textAlign: 'right', fontSize: '0.8em',
    fontWeight: 600, fontVariantNumeric: 'tabular-nums', flexShrink: 0,
  },

  fleetNote: { fontSize: '0.85em', opacity: 0.6, marginTop: '0.5rem' },

  // Table row (unit economics)
  tableRow: {
    display: 'flex', justifyContent: 'space-between',
    padding: '4px 0', fontSize: '0.9em',
  },

  // Sensitivity
  sensitivityList: { display: 'flex', flexDirection: 'column', gap: '6px' },
  sensitivityRow: {
    display: 'flex', alignItems: 'center', gap: '8px',
    padding: '6px 8px', borderRadius: '6px', cursor: 'pointer',
    transition: 'background-color 0.1s',
  },
  sensitivityPrice: { width: '70px', fontSize: '0.9em', fontWeight: 600, flexShrink: 0 },
  sensitivityBarTrack: { flex: 1, height: '22px', borderRadius: '3px' },
  sensitivityValue: {
    width: '80px', textAlign: 'right', fontSize: '0.85em',
    fontWeight: 600, fontVariantNumeric: 'tabular-nums', flexShrink: 0,
  },
  sensitivityMargin: {
    width: '40px', textAlign: 'right', fontSize: '0.75em',
    opacity: 0.5, flexShrink: 0,
  },

  // Notes
  notesGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '12px',
  },
  noteCard: {
    padding: '16px', borderRadius: '8px',
    border: '1px solid #333', backgroundColor: '#1a1a1a',
  },
  noteTitle: { fontWeight: 700, marginBottom: '6px', fontSize: '0.95em' },
  noteBody: { fontSize: '0.85em', opacity: 0.7, lineHeight: 1.5 },
}
