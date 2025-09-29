// Unit tests for frontend

import { render, screen } from '@testing-library/react'
import App from '../frontend/src/App'

test('renders hello message', () => {
    render(<App />)
    const linkElement = screen.getByText(/Hello Forecasting Dashboard/i)
    expect(linkElement).toBeInTheDocument()
})