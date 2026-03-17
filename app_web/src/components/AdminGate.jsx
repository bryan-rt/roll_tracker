import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'

const ADMIN_EMAIL = import.meta.env.VITE_ADMIN_EMAIL

export default function AdminGate({ children }) {
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [signingIn, setSigningIn] = useState(false)

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setLoading(false)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session)
        setLoading(false)
      }
    )

    return () => subscription.unsubscribe()
  }, [])

  const handleSignIn = async (e) => {
    e.preventDefault()
    setError(null)
    setSigningIn(true)
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) setError(error.message)
    setSigningIn(false)
  }

  const handleSignOut = async () => {
    await supabase.auth.signOut()
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <p>Loading…</p>
      </div>
    )
  }

  if (!session) {
    return (
      <div style={styles.container}>
        <div style={styles.card}>
          <h2 style={{ marginTop: 0 }}>Admin Sign In</h2>
          <form onSubmit={handleSignIn}>
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={styles.input}
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={styles.input}
            />
            {error && <p style={styles.error}>{error}</p>}
            <button type="submit" disabled={signingIn} style={styles.button}>
              {signingIn ? 'Signing in…' : 'Sign In'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  if (session.user.email !== ADMIN_EMAIL) {
    return (
      <div style={styles.container}>
        <div style={styles.card}>
          <h2 style={{ marginTop: 0 }}>Access Denied</h2>
          <p>You do not have admin access.</p>
          <button onClick={handleSignOut} style={styles.button}>Sign Out</button>
        </div>
      </div>
    )
  }

  return children
}

const styles = {
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 'calc(100vh - 48px)',
  },
  card: {
    padding: '2rem',
    borderRadius: '8px',
    border: '1px solid #333',
    maxWidth: '360px',
    width: '100%',
  },
  input: {
    display: 'block',
    width: '100%',
    padding: '0.5rem',
    marginBottom: '0.75rem',
    borderRadius: '4px',
    border: '1px solid #555',
    backgroundColor: '#1a1a1a',
    color: 'inherit',
    fontFamily: 'inherit',
    fontSize: '1rem',
    boxSizing: 'border-box',
  },
  button: {
    width: '100%',
    padding: '0.6em 1.2em',
    borderRadius: '8px',
    border: '1px solid transparent',
    backgroundColor: '#646cff',
    color: '#fff',
    fontSize: '1em',
    fontWeight: 500,
    fontFamily: 'inherit',
    cursor: 'pointer',
  },
  error: {
    color: '#ff6b6b',
    marginBottom: '0.75rem',
  },
}
