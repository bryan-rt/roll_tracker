import { Routes, Route, Link, useLocation } from 'react-router-dom'
import MatBlueprint from './pages/MatBlueprint'
import PricingDashboard from './pages/PricingDashboard'
import AdminGate from './components/AdminGate'

const NAV_LINKS = [
  { to: '/', label: 'Mat Blueprint' },
  { to: '/admin/pricing', label: 'Pricing Dashboard' },
]

function App() {
  const location = useLocation()

  return (
    <AdminGate>
      <div>
        <nav style={styles.nav}>
          <span style={styles.brand}>Roll Tracker</span>
          <div style={styles.links}>
            {NAV_LINKS.map(link => (
              <Link
                key={link.to}
                to={link.to}
                style={{
                  ...styles.link,
                  ...(location.pathname === link.to ? styles.linkActive : {}),
                }}
              >
                {link.label}
              </Link>
            ))}
          </div>
        </nav>
        <Routes>
          <Route path="/" element={<MatBlueprint />} />
          <Route path="/admin/pricing" element={<PricingDashboard />} />
        </Routes>
      </div>
    </AdminGate>
  )
}

const styles = {
  nav: {
    display: 'flex',
    alignItems: 'center',
    gap: '1.5rem',
    padding: '0 1rem',
    height: '48px',
    backgroundColor: '#1a1a1a',
    borderBottom: '1px solid #333',
  },
  brand: {
    fontWeight: 700,
    fontSize: '1rem',
    opacity: 0.6,
  },
  links: {
    display: 'flex',
    gap: '1rem',
  },
  link: {
    color: '#646cff',
    textDecoration: 'none',
    fontSize: '0.9rem',
    fontWeight: 500,
    padding: '0.25rem 0',
  },
  linkActive: {
    borderBottom: '2px solid #646cff',
  },
}

export default App
