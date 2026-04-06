import { useState, useEffect } from 'react'

const API_BASE = 'http://localhost:8000'

function App() {
  const [stats, setStats] = useState(null)
  const [patients, setPatients] = useState([])
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [patientReport, setPatientReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [reviewQueue, setReviewQueue] = useState([])
  const [analyzing, setAnalyzing] = useState(false)

  useEffect(() => {
    fetchDashboard()
  }, [])

  async function fetchDashboard() {
    try {
      const res = await fetch(`${API_BASE}/dashboard/stats`)
      const data = await res.json()
      setStats(data)
      
      const res2 = await fetch(`${API_BASE}/alerts/high-risk`)
      const alerts = await res2.json()
      setPatients(alerts)
    } catch (err) {
      console.error('Failed to fetch dashboard:', err)
    } finally {
      setLoading(false)
    }
  }

  async function analyzePatient(patientId) {
    setAnalyzing(true)
    try {
      const res = await fetch(`${API_BASE}/analyze/${patientId}`, { method: 'POST' })
      const data = await res.json()
      
      if (res.status === 202) {
        const threadId = data.detail.split('Thread ID: ')[1]
        setReviewQueue(prev => [...prev, { patientId, threadId }])
      } else if (res.ok) {
        setPatientReport(data)
        setSelectedPatient(patientId)
      }
    } catch (err) {
      console.error('Analysis failed:', err)
    } finally {
      setAnalyzing(false)
    }
  }

  async function approveReview(threadId) {
    try {
      await fetch(`${API_BASE}/review/${threadId}/approve`, { method: 'POST' })
      setReviewQueue(prev => prev.filter(r => r.threadId !== threadId))
      fetchDashboard()
    } catch (err) {
      console.error('Approval failed:', err)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl text-gray-600">Loading dashboard...</div>
      </div>
    )
  }

  const reduction = stats?.baseline_high_risk > 0 
    ? Math.round(((stats.baseline_high_risk - stats.high_risk_caught) / stats.baseline_high_risk) * 100) 
    : 0

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-slate-800">Oncology Clinical Safety Dashboard</h1>
        <p className="text-slate-500">LangGraph AI Decision Support System</p>
      </header>

      <main className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <StatCard label="Total Patients" value={stats?.total_patients || 0} color="blue" />
          <StatCard label="High Risk Detected" value={stats?.baseline_high_risk || 0} color="red" />
          <StatCard label="Risk Reduction" value={`${reduction}%`} color="green" />
          <StatCard label="Pending Reviews" value={reviewQueue.length} color="amber" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <h2 className="text-lg font-semibold mb-4">Risk Factor Distribution</h2>
              <RiskFactorChart factors={stats?.top_risk_factors || []} />
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">High Risk Patients (WBC &lt; 3.0)</h2>
              <PatientList 
                patients={patients} 
                onAnalyze={analyzePatient}
                selected={selectedPatient}
                analyzing={analyzing}
              />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Patient Detail</h2>
            {selectedPatient ? (
              <PatientDetail 
                report={patientReport} 
                patientId={selectedPatient}
              />
            ) : (
              <p className="text-slate-400">Select a patient to view details</p>
            )}
          </div>
        </div>

        {reviewQueue.length > 0 && (
          <div className="fixed bottom-4 right-4 bg-amber-50 border border-amber-200 rounded-lg shadow-lg p-4 max-w-sm">
            <h3 className="font-semibold text-amber-800 mb-2">Review Queue ({reviewQueue.length})</h3>
            {reviewQueue.map((item, i) => (
              <div key={i} className="flex items-center justify-between gap-2 mb-2">
                <span className="text-sm">{item.patientId}</span>
                <button
                  onClick={() => approveReview(item.threadId)}
                  className="px-3 py-1 bg-amber-600 text-white text-sm rounded hover:bg-amber-700"
                >
                  Approve
                </button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

function StatCard({ label, value, color }) {
  const colors = {
    blue: 'bg-blue-50 border-blue-200 text-blue-700',
    red: 'bg-red-50 border-red-200 text-red-700',
    green: 'bg-green-50 border-green-200 text-green-700',
    amber: 'bg-amber-50 border-amber-200 text-amber-700',
  }
  
  return (
    <div className={`rounded-lg border p-4 ${colors[color]}`}>
      <div className="text-sm opacity-75">{label}</div>
      <div className="text-3xl font-bold">{value}</div>
    </div>
  )
}

function RiskFactorChart({ factors }) {
  const maxCount = Math.max(...factors.map(f => f.count), 1)
  
  return (
    <div className="space-y-3">
      {factors.map((factor, i) => (
        <div key={i}>
          <div className="flex justify-between text-sm mb-1">
            <span className="font-medium">{factor.factor}</span>
            <span className="text-slate-500">{factor.count} patients</span>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-red-500 to-orange-400 h-3 rounded-full transition-all"
              style={{ width: `${(factor.count / maxCount) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

function PatientList({ patients, onAnalyze, selected, analyzing }) {
  if (patients.length === 0) {
    return <p className="text-slate-400">No high-risk patients</p>
  }
  
  return (
    <div className="space-y-2 max-h-64 overflow-y-auto">
      {patients.map((patient) => (
        <div 
          key={patient.Patient_ID}
          onClick={() => onAnalyze(patient.Patient_ID)}
          className={`p-3 rounded border cursor-pointer transition-all ${
            selected === patient.Patient_ID 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-slate-200 hover:border-slate-300'
          }`}
        >
          <div className="flex justify-between items-center">
            <div>
              <div className="font-medium">{patient.Patient_ID}</div>
              <div className="text-sm text-slate-500">{patient.Cancer_Type} | {patient.Age}yo</div>
            </div>
            <div className="text-right">
              <div className="text-red-600 font-bold">{patient.Latest_WBC.toFixed(1)}</div>
              <div className="text-xs text-slate-400">WBC</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function PatientDetail({ report, patientId }) {
  if (!report) return null
  
  const riskScore = report.risk_score || 0
  const getRiskColor = (score) => {
    if (score >= 0.65) return 'bg-red-500'
    if (score >= 0.4) return 'bg-orange-500'
    return 'bg-green-500'
  }
  
  return (
    <div>
      <div className="mb-4">
        <div className="text-sm text-slate-500 mb-1">Patient ID</div>
        <div className="font-semibold">{patientId}</div>
      </div>
      
      <div className="mb-4">
        <div className="text-sm text-slate-500 mb-1">Risk Score</div>
        <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden">
          <div 
            className={`absolute left-0 top-0 h-full ${getRiskColor(riskScore)} transition-all`}
            style={{ width: `${riskScore * 100}%` }}
          />
          <div className="absolute inset-0 flex items-center justify-center text-sm font-bold">
            {riskScore.toFixed(3)}
          </div>
        </div>
        <div className="flex justify-between text-xs text-slate-400 mt-1">
          <span>0.0</span>
          <span>0.4</span>
          <span>0.65</span>
          <span>1.0</span>
        </div>
      </div>
      
      <div className="mb-4">
        <div className="text-sm text-slate-500 mb-1">Risk Factors</div>
        <div className="flex flex-wrap gap-1">
          {report.risk_factors?.length > 0 ? (
            report.risk_factors.map((f, i) => (
              <span key={i} className="px-2 py-1 bg-red-100 text-red-700 text-xs rounded">
                {f}
              </span>
            ))
          ) : (
            <span className="text-slate-400 text-sm">None</span>
          )}
        </div>
      </div>
      
      <div className="mb-4">
        <div className="text-sm text-slate-500 mb-1">Clinical Summary</div>
        <div className="text-sm text-slate-700 bg-slate-50 p-2 rounded max-h-32 overflow-y-auto">
          {(() => {
            const parts = report.report?.split('CLINICAL SUMMARY:');
            // Take the last part if it was duplicated, otherwise take part [1]
            return parts?.length > 1 ? parts[parts.length - 1].split('NEXT BEST ACTION RECOMMENDATION:')[0].trim() : 'No summary available in report';
          })()}
        </div>
      </div>
      
      <div className="mb-4">
        <div className="text-sm text-slate-500 mb-1">Recommended Action</div>
        <div className="text-sm text-blue-700 bg-blue-50 p-2 rounded border border-blue-100 italic">
          {(() => {
            const parts = report.report?.split('NEXT BEST ACTION RECOMMENDATION:');
            return parts?.length > 1 ? parts[parts.length - 1].split('SAFETY AUDIT:')[0].trim() : 'N/A';
          })()}
        </div>
      </div>
      
      <div className="mb-4">
        <div className="text-sm text-slate-500 mb-1">Critic Notes</div>
        <div className="text-sm bg-amber-50 border border-amber-200 p-2 rounded">
          {report.report?.includes('CRITICAL') || report.report?.includes('WARNING') ? (
            <span className="text-amber-700">
              {(() => {
                const parts = report.report?.split('CRITIC NOTES:');
                return parts?.length > 1 ? parts[parts.length - 1].split('\n')[0].trim() : 'Review required';
              })()}
            </span>
          ) : (
            <span className="text-green-700">No critical flags</span>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
