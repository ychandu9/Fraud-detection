import { useState, useEffect, useCallback, useRef } from "react";
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  AreaChart, Area, RadialBarChart, RadialBar, LineChart, Line
} from "recharts";

const API = process.env.REACT_APP_API_URL || "https://fraudguard-api-0e2x.onrender.com";
const C = {
  safe:"#00f5a0", fraud:"#ff3860", warn:"#ffd166", accent:"#818cf8",
  bg:"#080b14", card:"#0f172a", border:"#1f2937",
  text:"#e2e8f0", muted:"#64748b",
  critical:"#ff073a", high:"#ff6b35", medium:"#ffd166", low:"#00f5a0",
};

const BANKS = ["-- Select Bank --","State Bank of India","HDFC Bank","ICICI Bank","Axis Bank","Kotak Mahindra Bank","Punjab National Bank","Bank of Baroda","Canara Bank","Union Bank of India","IndusInd Bank","Yes Bank","IDFC First Bank","Federal Bank","South Indian Bank","Bank of India","Central Bank of India","UCO Bank","Indian Bank","Citibank","HSBC","Other"];
const PAYMENT_MODES = ["UPI","NEFT","RTGS","IMPS","Credit Card","Debit Card","Net Banking","Wallet","Cheque","ATM Withdrawal","POS Terminal","Other"];
const MERCHANT_CATEGORIES = [
  {value:"retail",label:"Retail / Shopping"},{value:"food",label:"Food & Dining"},
  {value:"travel",label:"Travel & Transport"},{value:"electronics",label:"Electronics"},
  {value:"healthcare",label:"Healthcare"},{value:"education",label:"Education"},
  {value:"utilities",label:"Utilities / Bills"},{value:"entertainment",label:"Entertainment"},
  {value:"fuel",label:"Fuel / Petrol"},{value:"grocery",label:"Grocery"},
  {value:"gambling",label:"Gambling ⚠️"},{value:"crypto",label:"Cryptocurrency ⚠️"},
  {value:"other",label:"Other"},
];

// ── TRANSLATIONS ─────────────────────────────────────────────
const T = {
  en:{ detect:"🔍 Detect", dashboard:"📊 Dashboard", history:"📋 History", model:"🤖 Model", analytics:"📈 Analytics" },
  hi:{ detect:"🔍 जांच",   dashboard:"📊 डैशबोर्ड",  history:"📋 इतिहास",  model:"🤖 मॉडल",  analytics:"📈 विश्लेषण" },
  te:{ detect:"🔍 గుర్తించు",dashboard:"📊 డాష్‌బోర్డ్",history:"📋 చరిత్ర", model:"🤖 మోడల్", analytics:"📈 విశ్లేషణ" },
};

// ── SMALL COMPONENTS ─────────────────────────────────────────
function RiskBadge({ level }) {
  const map = { LOW:{c:C.low,l:"LOW RISK"}, MEDIUM:{c:C.medium,l:"MEDIUM RISK"}, HIGH:{c:C.high,l:"HIGH RISK"}, CRITICAL:{c:C.critical,l:"CRITICAL"} };
  const {c,l} = map[level]||map.LOW;
  return <span style={{background:c+"22",color:c,border:`1px solid ${c}55`,borderRadius:4,padding:"3px 12px",fontSize:11,fontWeight:700,letterSpacing:2,fontFamily:"monospace",boxShadow:`0 0 8px ${c}44`}}>{l}</span>;
}

function StatCard({ label, value, sub, color=C.safe }) {
  return (
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:"18px 22px",flex:1,minWidth:130,borderTop:`3px solid ${color}`}}>
      <div style={{color:C.muted,fontSize:11,letterSpacing:2,marginBottom:5,textTransform:"uppercase"}}>{label}</div>
      <div style={{color,fontSize:24,fontWeight:800,fontFamily:"monospace"}}>{value}</div>
      {sub && <div style={{color:C.muted,fontSize:12,marginTop:3}}>{sub}</div>}
    </div>
  );
}

function MetricBar({ label, value, color=C.safe }) {
  return (
    <div style={{marginBottom:14}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
        <span style={{fontSize:13,color:C.muted}}>{label}</span>
        <span style={{fontSize:14,fontWeight:800,fontFamily:"monospace",color}}>{value}%</span>
      </div>
      <div style={{background:"#1f2937",borderRadius:4,height:8}}>
        <div style={{background:`linear-gradient(90deg,${color},${color}88)`,borderRadius:4,height:8,width:`${Math.min(value,100)}%`,transition:"width 1s ease",boxShadow:`0 0 8px ${color}66`}}/>
      </div>
    </div>
  );
}

function Field({ label, name, type="text", value, onChange, options, placeholder, required, hint }) {
  const base = {width:"100%",background:"#0a0e1a",border:`1px solid ${C.border}`,borderRadius:8,color:C.text,padding:"10px 14px",fontSize:14,outline:"none",fontFamily:"inherit",boxSizing:"border-box"};
  return (
    <div style={{marginBottom:13}}>
      {label && <label style={{display:"block",color:C.muted,fontSize:11,marginBottom:5,letterSpacing:1,textTransform:"uppercase"}}>{label}{required&&<span style={{color:C.fraud,marginLeft:3}}>*</span>}</label>}
      {type==="select"?(
        <select name={name} value={value} onChange={onChange} style={base}>
          {options.map(o=><option key={o.value||o} value={o.value||o}>{o.label||o}</option>)}
        </select>
      ):type==="checkbox"?(
        <label style={{display:"flex",alignItems:"center",gap:10,cursor:"pointer",padding:"6px 0"}}>
          <input type="checkbox" name={name} checked={value} onChange={onChange} style={{width:16,height:16,accentColor:C.safe}}/>
          <span style={{color:C.text,fontSize:14}}>{placeholder}</span>
        </label>
      ):type==="textarea"?(
        <textarea name={name} value={value} onChange={onChange} placeholder={placeholder} rows={2} style={{...base,resize:"vertical",lineHeight:1.5}}/>
      ):(
        <input type={type} name={name} value={value} onChange={onChange} placeholder={placeholder} style={base}/>
      )}
      {hint&&<div style={{color:C.muted,fontSize:11,marginTop:3}}>{hint}</div>}
    </div>
  );
}

function FraudGauge({ value }) {
  const data=[{name:"Safe",value:100-value,fill:C.safe},{name:"Fraud",value,fill:C.fraud}];
  const color=value>70?C.fraud:value>40?C.high:C.safe;
  return (
    <div style={{textAlign:"center"}}>
      <ResponsiveContainer width="100%" height={175}>
        <RadialBarChart cx="50%" cy="75%" innerRadius="60%" outerRadius="95%" startAngle={180} endAngle={0} data={data}>
          <RadialBar dataKey="value" cornerRadius={8}/>
        </RadialBarChart>
      </ResponsiveContainer>
      <div style={{marginTop:-46,fontSize:36,fontWeight:900,color,fontFamily:"monospace"}}>{value}%</div>
      <div style={{color:C.muted,fontSize:12,marginTop:3}}>Fraud Probability</div>
    </div>
  );
}

function AlertBanner({ result, onDismiss }) {
  const isFraud=result?.is_fraud; const color=isFraud?C.fraud:C.safe;
  return (
    <div style={{position:"fixed",top:24,right:24,zIndex:9999,background:isFraud?"#ff386018":"#00f5a018",border:`2px solid ${color}`,borderRadius:16,padding:"20px 26px",maxWidth:380,boxShadow:`0 0 50px ${color}55`,animation:"slideIn 0.4s ease"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
        <div style={{flex:1}}>
          <div style={{color,fontSize:18,fontWeight:900,marginBottom:8}}>{isFraud?"🚨 FRAUD DETECTED":"✅ TRANSACTION SAFE"}</div>
          {result.bank_name&&<div style={{color:C.muted,fontSize:12,marginBottom:3}}>🏦 {result.bank_name}</div>}
          {result.cardholder_name&&<div style={{color:C.muted,fontSize:12,marginBottom:3}}>👤 {result.cardholder_name}</div>}
          {result.transaction_id&&<div style={{color:C.muted,fontSize:12,fontFamily:"monospace",marginBottom:6}}>🔖 {result.transaction_id}</div>}
          <div style={{color:C.text,fontSize:13,marginBottom:8}}>Confidence: <b style={{color}}>{isFraud?result.fraud_probability:result.safe_probability}%</b></div>
          <RiskBadge level={result.risk_level}/>
          {result.risk_factors?.length>0&&<ul style={{color:C.muted,fontSize:12,marginTop:10,paddingLeft:16,lineHeight:1.9}}>{result.risk_factors.map((f,i)=><li key={i}>{f}</li>)}</ul>}
        </div>
        <button onClick={onDismiss} style={{background:"none",border:"none",color:C.muted,cursor:"pointer",fontSize:20,marginLeft:12,padding:0}}>✕</button>
      </div>
    </div>
  );
}

// ── FEATURE 4: LOGIN PAGE ─────────────────────────────────────
function LoginPage({ onLogin }) {
  const [mode,    setMode]    = useState("login");
  const [form,    setForm]    = useState({username:"",password:"",name:""});
  const [error,   setError]   = useState("");
  const [loading, setLoading] = useState(false);

  const handle = async() => {
    setError(""); setLoading(true);
    try {
      const endpoint = mode==="login" ? "/login" : "/register";
      const r = await fetch(`${API}${endpoint}`,{
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(form)
      });
      const data = await r.json();
      if(!r.ok) { setError(data.error||"Something went wrong"); setLoading(false); return; }
      localStorage.setItem("fg_token", data.token);
      localStorage.setItem("fg_name",  data.name);
      onLogin(data.name, data.token);
    } catch(e) { setError("Cannot reach backend. Make sure Flask is running."); }
    setLoading(false);
  };

  return (
    <div style={{minHeight:"100vh",background:C.bg,display:"flex",alignItems:"center",justifyContent:"center",fontFamily:"'DM Sans','Segoe UI',sans-serif"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700;800;900&family=Space+Mono:wght@700&display=swap'); *{box-sizing:border-box;margin:0;padding:0} @keyframes fadeUp{from{transform:translateY(30px);opacity:0}to{transform:translateY(0);opacity:1}}`}</style>
      <div style={{width:420,animation:"fadeUp 0.5s ease"}}>
        {/* Logo */}
        <div style={{textAlign:"center",marginBottom:32}}>
          <div style={{width:64,height:64,background:`linear-gradient(135deg,${C.safe},${C.accent})`,borderRadius:16,display:"flex",alignItems:"center",justifyContent:"center",fontSize:30,margin:"0 auto 16px"}}>🛡️</div>
          <div style={{fontWeight:900,fontSize:24,fontFamily:"'Space Mono',monospace",color:C.text}}>FraudGuard <span style={{color:C.safe}}>AI</span></div>
          <div style={{color:C.muted,fontSize:12,marginTop:4,letterSpacing:2}}>ML-POWERED FRAUD DETECTION</div>
        </div>

        {/* Card */}
        <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:16,padding:32,boxShadow:`0 0 60px #00f5a011`}}>
          {/* Tabs */}
          <div style={{display:"flex",background:"#080b14",borderRadius:10,padding:4,marginBottom:24}}>
            {["login","register"].map(m=>(
              <button key={m} onClick={()=>{setMode(m);setError("");}} style={{flex:1,padding:"10px",border:"none",borderRadius:8,cursor:"pointer",fontWeight:700,fontSize:14,fontFamily:"inherit",background:mode===m?`linear-gradient(90deg,${C.safe}33,${C.accent}33)`:"none",color:mode===m?C.safe:C.muted,transition:"all 0.2s",textTransform:"capitalize"}}>
                {m==="login"?"🔐 Login":"📝 Register"}
              </button>
            ))}
          </div>

          {mode==="register"&&(
            <div style={{marginBottom:16}}>
              <label style={{display:"block",color:C.muted,fontSize:11,marginBottom:5,letterSpacing:1,textTransform:"uppercase"}}>Full Name *</label>
              <input value={form.name} onChange={e=>setForm(f=>({...f,name:e.target.value}))} placeholder="e.g. Rahul Sharma"
                style={{width:"100%",background:"#0a0e1a",border:`1px solid ${C.border}`,borderRadius:8,color:C.text,padding:"11px 14px",fontSize:14,outline:"none",fontFamily:"inherit"}}/>
            </div>
          )}
          <div style={{marginBottom:16}}>
            <label style={{display:"block",color:C.muted,fontSize:11,marginBottom:5,letterSpacing:1,textTransform:"uppercase"}}>Username *</label>
            <input value={form.username} onChange={e=>setForm(f=>({...f,username:e.target.value}))} placeholder="e.g. rahul123"
              style={{width:"100%",background:"#0a0e1a",border:`1px solid ${C.border}`,borderRadius:8,color:C.text,padding:"11px 14px",fontSize:14,outline:"none",fontFamily:"inherit"}}/>
          </div>
          <div style={{marginBottom:20}}>
            <label style={{display:"block",color:C.muted,fontSize:11,marginBottom:5,letterSpacing:1,textTransform:"uppercase"}}>Password *</label>
            <input type="password" value={form.password} onChange={e=>setForm(f=>({...f,password:e.target.value}))}
              onKeyDown={e=>e.key==="Enter"&&handle()}
              placeholder="Min 6 characters"
              style={{width:"100%",background:"#0a0e1a",border:`1px solid ${C.border}`,borderRadius:8,color:C.text,padding:"11px 14px",fontSize:14,outline:"none",fontFamily:"inherit"}}/>
          </div>

          {error&&<div style={{background:"#ff386015",border:`1px solid ${C.fraud}44`,borderRadius:8,padding:"10px 14px",color:C.fraud,fontSize:13,marginBottom:16}}>⚠️ {error}</div>}

          <button onClick={handle} disabled={loading} style={{width:"100%",padding:"14px",background:loading?C.muted:`linear-gradient(90deg,${C.safe},#00c896)`,border:"none",borderRadius:10,color:"#080b14",fontWeight:900,fontSize:15,cursor:loading?"not-allowed":"pointer",fontFamily:"inherit",boxShadow:loading?"none":`0 0 24px ${C.safe}44`}}>
            {loading?"⏳ Please wait...":(mode==="login"?"🔐 Login":"📝 Create Account")}
          </button>

          {/* Demo account hint */}
          <div style={{marginTop:16,padding:"10px 14px",background:"#818cf811",borderRadius:8,fontSize:12,color:C.muted,textAlign:"center"}}>
            💡 First time? Click <b style={{color:C.accent}}>Register</b> to create your account
          </div>
        </div>
      </div>
    </div>
  );
}

// ── FEATURE 1: CHATBOT ────────────────────────────────────────
// function ChatBot({ lang }) {
//   const [messages, setMessages] = useState([
//     {role:"assistant",content:"👋 Hi! I'm your FraudGuard AI Assistant.\n\nAsk me anything about:\n• UPI scams & phishing\n• OTP fraud & card skimming\n• What to do if scammed\n• How FraudGuard AI works\n\n**Emergency Helpline: 1930**"}
//   ]);
//   const [input,   setInput]   = useState("");
//   const [loading, setLoading] = useState(false);
//   const bottomRef = useRef(null);
//   useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:"smooth"}); },[messages]);

//   const sendMessage = async() => {
//     if(!input.trim()) return;
//     const userMsg = {role:"user",content:input};
//     setMessages(m=>[...m,userMsg]); setInput(""); setLoading(true);
//     try {
//       const r = await fetch(`${API}/chat`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:input,history:messages,language:lang})});
//       const data = await r.json();
//       setMessages(m=>[...m,{role:"assistant",content:data.reply||"Sorry, I couldn't process that."}]);
//     } catch(e) { setMessages(m=>[...m,{role:"assistant",content:"⚠️ Cannot reach backend."}]); }
//     setLoading(false);
//   };

//   const quickQs = ["What is UPI fraud?","My OTP was stolen!","How to identify phishing?","What is card skimming?","I got scammed, what to do?","How does FraudGuard AI work?"];

//   const renderMsg = (content) => content.split('\n').map((line,i)=>{
//     if(line.startsWith('**')&&line.endsWith('**')) return <div key={i} style={{fontWeight:800,color:C.text,marginTop:6}}>{line.replace(/\*\*/g,'')}</div>;
//     if(line.startsWith('• ')||line.startsWith('- ')) return <div key={i} style={{color:C.muted,fontSize:13,marginLeft:8,marginTop:2}}>• {line.slice(2)}</div>;
//     if(line.match(/^\d+\./)) return <div key={i} style={{color:C.muted,fontSize:13,marginLeft:8,marginTop:2}}>{line}</div>;
//     return line?<div key={i} style={{marginTop:i>0?3:0}}>{line}</div>:<div key={i} style={{height:4}}/>;
//   });

//   return (
//     <div style={{display:"grid",gridTemplateColumns:"1fr 260px",gap:20,height:"calc(100vh - 190px)"}}>
//       <div style={{display:"flex",flexDirection:"column",background:C.card,border:`1px solid ${C.border}`,borderRadius:14,overflow:"hidden"}}>
//         <div style={{padding:"14px 22px",borderBottom:`1px solid ${C.border}`,display:"flex",alignItems:"center",gap:12}}>
//           <div style={{width:38,height:38,background:`linear-gradient(135deg,${C.accent},${C.safe})`,borderRadius:10,display:"flex",alignItems:"center",justifyContent:"center",fontSize:18}}>🤖</div>
//           <div>
//             <div style={{fontWeight:800,fontSize:14}}>FraudGuard AI Assistant</div>
//             <div style={{color:C.safe,fontSize:11,display:"flex",alignItems:"center",gap:5}}><span style={{width:6,height:6,borderRadius:"50%",background:C.safe,display:"inline-block",animation:"pulse 2s infinite"}}/>Powered by Claude AI</div>
//           </div>
//         </div>
//         <div style={{flex:1,overflowY:"auto",padding:"18px 22px",display:"flex",flexDirection:"column",gap:14}}>
//           {messages.map((m,i)=>(
//             <div key={i} style={{display:"flex",justifyContent:m.role==="user"?"flex-end":"flex-start"}}>
//               <div style={{maxWidth:"82%",padding:"11px 15px",borderRadius:m.role==="user"?"14px 14px 4px 14px":"14px 14px 14px 4px",background:m.role==="user"?`linear-gradient(135deg,${C.accent}44,${C.safe}22)`:C.bg,border:`1px solid ${m.role==="user"?C.accent:C.border}`,fontSize:14,lineHeight:1.7,color:C.text}}>
//                 {m.role==="assistant"&&<div style={{color:C.safe,fontSize:10,fontWeight:700,marginBottom:5,letterSpacing:1}}>🤖 FRAUDGUARD AI</div>}
//                 {renderMsg(m.content)}
//               </div>
//             </div>
//           ))}
//           {loading&&<div style={{display:"flex",justifyContent:"flex-start"}}><div style={{padding:"11px 15px",borderRadius:"14px 14px 14px 4px",background:C.bg,border:`1px solid ${C.border}`}}><div style={{display:"flex",gap:4}}>{[0,1,2].map(i=><div key={i} style={{width:7,height:7,borderRadius:"50%",background:C.safe,animation:`pulse 1s ${i*0.2}s infinite`}}/>)}</div></div></div>}
//           <div ref={bottomRef}/>
//         </div>
//         <div style={{padding:"14px 22px",borderTop:`1px solid ${C.border}`,display:"flex",gap:10}}>
//           <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&sendMessage()}
//             placeholder="Ask about fraud, UPI scams, phishing..."
//             style={{flex:1,background:"#0a0e1a",border:`1px solid ${C.border}`,borderRadius:10,color:C.text,padding:"11px 14px",fontSize:14,outline:"none",fontFamily:"inherit"}}/>
//           <button onClick={sendMessage} disabled={loading||!input.trim()} style={{background:`linear-gradient(135deg,${C.safe},#00c896)`,border:"none",borderRadius:10,color:"#080b14",fontWeight:900,fontSize:16,cursor:"pointer",width:46,height:44}}>➤</button>
//         </div>
//       </div>
//       <div style={{display:"flex",flexDirection:"column",gap:12}}>
//         <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:18}}>
//           <div style={{color:C.muted,fontSize:11,letterSpacing:2,marginBottom:12,textTransform:"uppercase"}}>⚡ Quick Questions</div>
//           {quickQs.map((q,i)=>(
//             <button key={i} onClick={()=>setInput(q)} style={{display:"block",width:"100%",textAlign:"left",background:"#0a0e1a",border:`1px solid ${C.border}`,borderRadius:8,color:C.text,padding:"9px 12px",fontSize:12,cursor:"pointer",marginBottom:7,fontFamily:"inherit",lineHeight:1.4,transition:"border-color 0.2s"}}
//               onMouseEnter={e=>e.currentTarget.style.borderColor=C.safe} onMouseLeave={e=>e.currentTarget.style.borderColor=C.border}>{q}</button>
//           ))}
//         </div>
//         <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:18}}>
//           <div style={{color:C.muted,fontSize:11,letterSpacing:2,marginBottom:10,textTransform:"uppercase"}}>🆘 Emergency</div>
//           <div style={{fontSize:13,color:C.text,marginBottom:6}}>Cyber Crime Helpline</div>
//           <div style={{fontSize:28,fontWeight:900,color:C.fraud,fontFamily:"monospace",marginBottom:6}}>1930</div>
//           <div style={{fontSize:11,color:C.muted,marginBottom:10}}>Available 24/7</div>
//           <a href="https://cybercrime.gov.in" target="_blank" rel="noreferrer" style={{color:C.accent,fontSize:12,textDecoration:"none"}}>🌐 cybercrime.gov.in</a>
//         </div>
//       </div>
//     </div>
//   );
// }

// ── FEATURE 2: ADVANCED ANALYTICS ────────────────────────────
function AdvancedAnalytics({ stats, history }) {
  if(!stats||!history.length) return <div style={{textAlign:"center",color:C.muted,padding:80}}>No data yet. Analyze some transactions first.</div>;

  // Hourly distribution
  const hourData = Array.from({length:24},(_,i)=>({hour:`${i}:00`,count:0,fraud:0}));
  history.forEach(t=>{
    try{
      const h = new Date(t.timestamp).getHours();
      hourData[h].count++;
      if(t.is_fraud) hourData[h].fraud++;
    }catch{}
  });

  // Payment mode fraud rate
  const modeMap = {};
  history.forEach(t=>{
    const m = t.payment_mode||"Unknown";
    if(!modeMap[m]) modeMap[m]={total:0,fraud:0};
    modeMap[m].total++;
    if(t.is_fraud) modeMap[m].fraud++;
  });
  const modeData = Object.entries(modeMap).map(([k,v])=>({name:k,total:v.total,fraud:v.fraud,rate:v.total>0?Math.round(v.fraud/v.total*100):0})).sort((a,b)=>b.rate-a.rate);

  // Amount range breakdown
  const ranges = [{label:"<500",min:0,max:500},{label:"500-2K",min:500,max:2000},{label:"2K-10K",min:2000,max:10000},{label:"10K-50K",min:10000,max:50000},{label:">50K",min:50000,max:Infinity}];
  const amtData = ranges.map(r=>{
    const txns = history.filter(t=>{ const a=parseFloat(t.amount||0); return a>=r.min&&a<r.max; });
    return {name:r.label,total:txns.length,fraud:txns.filter(t=>t.is_fraud).length};
  });

  // Bank fraud breakdown
  const bankMap = {};
  history.forEach(t=>{
    const b = t.bank_name||"Unknown";
    if(!bankMap[b]) bankMap[b]={total:0,fraud:0};
    bankMap[b].total++;
    if(t.is_fraud) bankMap[b].fraud++;
  });
  const bankData = Object.entries(bankMap).map(([k,v])=>({name:k.split(" ").slice(0,2).join(" "),total:v.total,fraud:v.fraud})).sort((a,b)=>b.fraud-a.fraud).slice(0,6);

  // Top risk factors
  const rfMap = {};
  history.forEach(t=>{ (t.risk_factors||[]).forEach(rf=>{ rfMap[rf]=(rfMap[rf]||0)+1; }); });
  const rfData = Object.entries(rfMap).map(([k,v])=>({name:k.replace("🚨 CRITICAL: ","").replace("Transaction from ","").slice(0,35),count:v})).sort((a,b)=>b.count-a.count).slice(0,6);

  // Fraud trend (last 15)
  const trendData = history.slice(-15).map((t,i)=>({n:`#${i+1}`,score:t.fraud_probability,ml:t.ml_score||0,boost:t.rule_boost||0}));

  const chartStyle = {background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:22};
  const ttStyle    = {background:C.card,border:`1px solid ${C.border}`,borderRadius:8};

  return (
    <div>
      <h2 style={{marginBottom:22,fontSize:19,fontWeight:800}}>📈 Advanced Analytics</h2>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>

        {/* Hourly heatmap */}
        <div style={{...chartStyle,gridColumn:"span 2"}}>
          <div style={{fontWeight:700,marginBottom:14}}>🕐 Hourly Transaction Heatmap (All 24 Hours)</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={hourData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
              <XAxis dataKey="hour" tick={{fill:C.muted,fontSize:9}} interval={1}/>
              <YAxis tick={{fill:C.muted,fontSize:10}}/>
              <Tooltip contentStyle={ttStyle}/>
              <Legend formatter={v=><span style={{color:C.text,fontSize:12}}>{v}</span>}/>
              <Bar dataKey="count" name="Total"  fill={C.accent} radius={[3,3,0,0]}/>
              <Bar dataKey="fraud" name="Fraud"  fill={C.fraud}  radius={[3,3,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
          <div style={{color:C.muted,fontSize:11,marginTop:8,textAlign:"center"}}>Peak fraud hours highlighted in red — transactions at unusual hours (0-5 AM) are highest risk</div>
        </div>

        {/* Payment mode fraud rate */}
        <div style={chartStyle}>
          <div style={{fontWeight:700,marginBottom:14}}>💳 Payment Mode Fraud Rate</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={modeData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
              <XAxis type="number" tick={{fill:C.muted,fontSize:10}}/>
              <YAxis dataKey="name" type="category" tick={{fill:C.muted,fontSize:10}} width={80}/>
              <Tooltip contentStyle={ttStyle} formatter={(v,n)=>[`${v}${n==="rate"?"%":""}`,n]}/>
              <Bar dataKey="total" name="Total"     fill={C.accent} radius={[0,3,3,0]}/>
              <Bar dataKey="fraud" name="Fraud"     fill={C.fraud}  radius={[0,3,3,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Amount range */}
        <div style={chartStyle}>
          <div style={{fontWeight:700,marginBottom:14}}>💰 Fraud by Amount Range</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={amtData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
              <XAxis dataKey="name" tick={{fill:C.muted,fontSize:11}}/>
              <YAxis tick={{fill:C.muted,fontSize:10}}/>
              <Tooltip contentStyle={ttStyle}/>
              <Legend formatter={v=><span style={{color:C.text,fontSize:12}}>{v}</span>}/>
              <Bar dataKey="total" name="Total" fill={C.accent} radius={[4,4,0,0]}/>
              <Bar dataKey="fraud" name="Fraud" fill={C.fraud}  radius={[4,4,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Bank fraud breakdown */}
        <div style={chartStyle}>
          <div style={{fontWeight:700,marginBottom:14}}>🏦 Bank-wise Fraud Breakdown</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={bankData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
              <XAxis dataKey="name" tick={{fill:C.muted,fontSize:9}}/>
              <YAxis tick={{fill:C.muted,fontSize:10}}/>
              <Tooltip contentStyle={ttStyle}/>
              <Legend formatter={v=><span style={{color:C.text,fontSize:12}}>{v}</span>}/>
              <Bar dataKey="total" name="Total" fill={C.accent} radius={[4,4,0,0]}/>
              <Bar dataKey="fraud" name="Fraud" fill={C.fraud}  radius={[4,4,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top risk factors */}
        <div style={chartStyle}>
          <div style={{fontWeight:700,marginBottom:14}}>⚠️ Top Risk Factors Detected</div>
          {rfData.length>0?(
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={rfData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
                <XAxis type="number" tick={{fill:C.muted,fontSize:10}}/>
                <YAxis dataKey="name" type="category" tick={{fill:C.muted,fontSize:9}} width={140}/>
                <Tooltip contentStyle={ttStyle}/>
                <Bar dataKey="count" name="Count" fill={C.high} radius={[0,4,4,0]}/>
              </BarChart>
            </ResponsiveContainer>
          ):<div style={{color:C.muted,textAlign:"center",paddingTop:60}}>No risk factors yet</div>}
        </div>

        {/* ML score vs Rule boost trend */}
        <div style={{...chartStyle,gridColumn:"span 2"}}>
          <div style={{fontWeight:700,marginBottom:14}}>🧠 ML Score vs Rule Boost (Last 15 Transactions)</div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={trendData}>
              <defs>
                <linearGradient id="ga1" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.accent} stopOpacity={0.3}/><stop offset="95%" stopColor={C.accent} stopOpacity={0}/></linearGradient>
                <linearGradient id="ga2" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.warn}   stopOpacity={0.3}/><stop offset="95%" stopColor={C.warn}   stopOpacity={0}/></linearGradient>
                <linearGradient id="ga3" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.fraud}  stopOpacity={0.3}/><stop offset="95%" stopColor={C.fraud}  stopOpacity={0}/></linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
              <XAxis dataKey="n" tick={{fill:C.muted,fontSize:10}}/>
              <YAxis tick={{fill:C.muted,fontSize:10}}/>
              <Tooltip contentStyle={ttStyle} formatter={(v,n)=>[`${v}%`,n]}/>
              <Legend formatter={v=><span style={{color:C.text,fontSize:12}}>{v}</span>}/>
              <Area type="monotone" dataKey="ml"    name="ML Score"    stroke={C.accent} fill="url(#ga1)" strokeWidth={2}/>
              <Area type="monotone" dataKey="boost" name="Rule Boost"  stroke={C.warn}   fill="url(#ga2)" strokeWidth={2}/>
              <Area type="monotone" dataKey="score" name="Final Score" stroke={C.fraud}  fill="url(#ga3)" strokeWidth={2}/>
            </AreaChart>
          </ResponsiveContainer>
          <div style={{color:C.muted,fontSize:11,marginTop:8,textAlign:"center"}}>Shows how ML model score and rule-based boost combine to give the final fraud probability</div>
        </div>

      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
//  MAIN APP
// ══════════════════════════════════════════════════════════════
export default function App() {
  const [user,    setUser]    = useState(null);
  const [token,   setToken]   = useState(null);
  const [tab,     setTab]     = useState("detect");
  const [lang,    setLang]    = useState("en");
  const t = T[lang]||T.en;

  const [form, setForm] = useState({
    cardholder_name:"",sender_account:"",sender_ifsc:"",bank_name:"",
    receiver_name:"",receiver_account:"",receiver_bank:"",receiver_ifsc:"",
    transaction_id:"",amount:"",payment_mode:"UPI",
    transaction_type:"purchase",merchant_category:"retail",
    merchant_name:"",device_type:"mobile",transaction_note:"",
    hour:new Date().getHours().toString(),
    location_mismatch:false,new_merchant:false,high_frequency:false,
    online_transaction:false,card_present:true,international:false,
    saved_beneficiary:true,first_time_large_amt:false,
    suspicious_link:false,otp_shared:false,
  });

  const [result,    setResult]    = useState(null);
  const [alertData, setAlertData] = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [pdfLoading,setPdfLoading]= useState(false);
  const [history,   setHistory]   = useState([]);
  const [stats,     setStats]     = useState(null);
  const [health,    setHealth]    = useState(null);
  const [error,     setError]     = useState(null);
  const [formError, setFormError] = useState("");

  // Check saved login on mount
  useEffect(()=>{
    const savedToken = localStorage.getItem("fg_token");
    const savedName  = localStorage.getItem("fg_name");
    if(savedToken&&savedName){
      fetch(`${API}/verify_token`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({token:savedToken})})
        .then(r=>r.ok?r.json():null)
        .then(data=>{ if(data?.valid){ setUser(savedName); setToken(savedToken); } })
        .catch(()=>{});
    }
  },[]);

  const fetchHistory = useCallback(async()=>{ try{ const r=await fetch(`${API}/history`); setHistory(await r.json()); }catch{} },[]);
  const fetchStats   = useCallback(async()=>{ try{ const r=await fetch(`${API}/stats`);   setStats(await r.json()); }catch{} },[]);
  const fetchHealth  = useCallback(async()=>{ try{ const r=await fetch(`${API}/health`);  setHealth(await r.json()); }catch{} },[]);

  useEffect(()=>{
    if(!user) return;
    fetchHealth(); fetchHistory(); fetchStats();
    const i=setInterval(()=>{ fetchHealth(); fetchHistory(); fetchStats(); },15000);
    return ()=>clearInterval(i);
  },[user,fetchHealth,fetchHistory,fetchStats]);

  const handleLogin  = (name, tok) => { setUser(name); setToken(tok); };
  const handleLogout = () => {
    fetch(`${API}/logout`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({token})});
    localStorage.removeItem("fg_token"); localStorage.removeItem("fg_name");
    setUser(null); setToken(null); setResult(null);
  };

  const handleChange = e=>{
    const{name,value,type,checked}=e.target;
    setForm(f=>({...f,[name]:type==="checkbox"?checked:value}));
    setFormError("");
  };

  const validate = ()=>{
    if(!form.cardholder_name.trim())  return "Sender Name is required";
    if(!form.sender_account.trim())   return "Sender Account Number is required";
    if(!form.bank_name||form.bank_name==="-- Select Bank --") return "Please select your Bank";
    if(!form.receiver_name.trim())    return "Receiver Name is required";
    if(!form.receiver_account.trim()) return "Receiver Account Number is required";
    if(!form.transaction_id.trim())   return "Transaction / Reference ID is required";
    if(!form.amount||parseFloat(form.amount)<=0) return "Enter a valid amount";
    return "";
  };

  const handleSubmit = async()=>{
    const err=validate(); if(err){ setFormError(err); return; }
    setLoading(true); setFormError("");
    try{
      const r=await fetch(`${API}/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(form)});
      const data=await r.json();
      if(data.error) throw new Error(data.error);
      const enriched={...data,
        transaction_id:form.transaction_id,bank_name:form.bank_name,
        cardholder_name:form.cardholder_name,sender_account:form.sender_account,
        receiver_name:form.receiver_name,receiver_account:form.receiver_account,
        amount:form.amount,payment_mode:form.payment_mode,
      };
      setResult(enriched); setAlertData(enriched);
      fetchHistory(); fetchStats();
    }catch(e){ setError(`❌ ${e.message||"Cannot reach backend."}`); }
    setLoading(false);
  };

  // ── FEATURE 1: PDF DOWNLOAD ───────────────────────────────
  const handleDownloadPDF = async()=>{
    if(!result) return;
    setPdfLoading(true);
    try{
      const r = await fetch(`${API}/generate_pdf`,{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({result, form})
      });
      if(!r.ok) throw new Error("PDF generation failed");
      const blob = await r.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `FraudReport_${result.transaction_id||"TXN"}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    }catch(e){ setError("❌ PDF failed. Run: pip install reportlab"); }
    setPdfLoading(false);
  };

  const handleRetrain = async()=>{
    setLoading(true);
    try{ await fetch(`${API}/retrain`,{method:"POST"}); fetchStats(); fetchHealth(); }catch{}
    setLoading(false);
  };

  const handleReset = ()=>{
    setForm(f=>({...f,cardholder_name:"",sender_account:"",sender_ifsc:"",receiver_name:"",receiver_account:"",receiver_bank:"",receiver_ifsc:"",transaction_id:"",amount:"",merchant_name:"",transaction_note:"",location_mismatch:false,new_merchant:false,high_frequency:false,online_transaction:false,card_present:true,international:false,saved_beneficiary:true,first_time_large_amt:false,suspicious_link:false,otp_shared:false}));
    setResult(null); setFormError("");
  };

  const TABS = ["detect","dashboard","analytics","history","model"];

  const SectionHeader = ({icon,title})=>(
    <div style={{display:"flex",alignItems:"center",gap:8,color:C.muted,fontSize:11,letterSpacing:2,marginBottom:14,textTransform:"uppercase",paddingBottom:10,borderBottom:`1px solid ${C.border}`}}>
      <span>{icon}</span><span>{title}</span>
    </div>
  );

  // ── SHOW LOGIN IF NOT AUTHENTICATED ──────────────────────
  if(!user) return <LoginPage onLogin={handleLogin}/>;

  return (
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,fontFamily:"'DM Sans','Segoe UI',sans-serif"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800;900&family=Space+Mono:wght@400;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:${C.bg}}::-webkit-scrollbar-thumb{background:${C.border};border-radius:3px}
        @keyframes slideIn{from{transform:translateX(120%);opacity:0}to{transform:translateX(0);opacity:1}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
        select,input,textarea{transition:border-color 0.2s,box-shadow 0.2s}
        select:focus,input:focus,textarea:focus{border-color:${C.safe}!important;box-shadow:0 0 0 3px ${C.safe}22!important}
      `}</style>

      {alertData&&<AlertBanner result={alertData} onDismiss={()=>setAlertData(null)}/>}
      {error&&(
        <div style={{position:"fixed",top:24,right:24,zIndex:9999,background:"#ff386015",border:`1.5px solid ${C.fraud}`,borderRadius:14,padding:"14px 22px",maxWidth:380,boxShadow:`0 0 30px ${C.fraud}44`,animation:"slideIn 0.4s ease",display:"flex",justifyContent:"space-between",alignItems:"center",gap:16}}>
          <div style={{color:C.fraud,fontSize:13,fontWeight:700}}>{error}</div>
          <button onClick={()=>setError(null)} style={{background:"none",border:"none",color:C.muted,cursor:"pointer",fontSize:18,padding:0}}>✕</button>
        </div>
      )}

      {/* HEADER */}
      <header style={{borderBottom:`1px solid ${C.border}`,padding:"13px 26px",display:"flex",alignItems:"center",justifyContent:"space-between",position:"sticky",top:0,background:C.bg+"ee",backdropFilter:"blur(14px)",zIndex:100}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{width:38,height:38,background:`linear-gradient(135deg,${C.safe},${C.accent})`,borderRadius:10,display:"flex",alignItems:"center",justifyContent:"center",fontSize:18}}>🛡️</div>
          <div>
            <div style={{fontWeight:900,fontSize:17,fontFamily:"'Space Mono',monospace",letterSpacing:-0.5}}>FraudGuard <span style={{color:C.safe}}>AI</span></div>
            <div style={{fontSize:10,color:C.muted,letterSpacing:1}}>KAGGLE DATASET · ENSEMBLE ML · REAL-TIME</div>
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:18}}>
          {/* Language */}
          <div style={{display:"flex",gap:5}}>
            {[["en","EN"],["hi","हि"],["te","తె"]].map(([code,label])=>(
              <button key={code} onClick={()=>setLang(code)} style={{padding:"4px 9px",borderRadius:6,fontSize:11,fontWeight:700,cursor:"pointer",background:lang===code?C.safe+"22":"none",border:`1px solid ${lang===code?C.safe:C.border}`,color:lang===code?C.safe:C.muted,transition:"all 0.2s",fontFamily:"inherit"}}>{label}</button>
            ))}
          </div>
          {health&&(
            <div style={{display:"flex",flexDirection:"column",alignItems:"flex-end",gap:2}}>
              <div style={{display:"flex",alignItems:"center",gap:6,fontSize:12}}>
                <span style={{width:7,height:7,borderRadius:"50%",background:C.safe,display:"inline-block",animation:"pulse 2s infinite"}}/>
                <span style={{color:C.muted}}>API Online</span>
              </div>
              <div style={{fontSize:10,color:C.muted,fontFamily:"monospace"}}>AUC {health.auc_roc}% · F1 {health.f1_fraud}%</div>
            </div>
          )}
          {stats&&<div style={{color:C.muted,fontSize:12}}><span style={{color:C.safe,fontWeight:700}}>{stats.total}</span> analyzed</div>}
          {/* User + Logout */}
          <div style={{display:"flex",alignItems:"center",gap:10,padding:"6px 14px",background:C.card,border:`1px solid ${C.border}`,borderRadius:10}}>
            <span style={{fontSize:16}}>👤</span>
            <span style={{fontSize:13,fontWeight:700,color:C.text}}>{user}</span>
            <button onClick={handleLogout} style={{background:"none",border:`1px solid ${C.border}`,color:C.muted,borderRadius:6,padding:"3px 10px",fontSize:11,cursor:"pointer",fontFamily:"inherit"}}>Logout</button>
          </div>
        </div>
      </header>

      {/* TABS */}
      <div style={{padding:"0 26px",borderBottom:`1px solid ${C.border}`,display:"flex",overflowX:"auto"}}>
        {TABS.map(tb=>(
          <button key={tb} onClick={()=>setTab(tb)} style={{background:"none",border:"none",cursor:"pointer",padding:"12px 16px",fontSize:13,fontWeight:600,color:tab===tb?C.safe:C.muted,borderBottom:tab===tb?`2px solid ${C.safe}`:"2px solid transparent",transition:"all 0.2s",fontFamily:"inherit",whiteSpace:"nowrap"}}>
            {t[tb]||tb}
          </button>
        ))}
      </div>

      <main style={{padding:26,maxWidth:1400,margin:"0 auto"}}>

        {/* ══ DETECT ══ */}
        {tab==="detect"&&(
          <div style={{display:"grid",gridTemplateColumns:"1.1fr 0.9fr",gap:22}}>
            <div>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16}}>
                <h2 style={{fontSize:18,fontWeight:800}}>Transaction Details</h2>
                <button onClick={handleReset} style={{background:"none",border:`1px solid ${C.border}`,color:C.muted,borderRadius:8,padding:"5px 12px",fontSize:12,cursor:"pointer",fontFamily:"inherit"}}>🔄 Reset</button>
              </div>
              <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:22,display:"flex",flexDirection:"column",gap:18}}>
                <div>
                  <SectionHeader icon="👤" title="Sender / Your Details"/>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 16px"}}>
                    <Field label="Full Name"        name="cardholder_name" value={form.cardholder_name} onChange={handleChange} placeholder="e.g. Rahul Sharma" required/>
                    <Field label="Your Bank"        name="bank_name" type="select" value={form.bank_name} onChange={handleChange} options={BANKS.map(b=>({value:b,label:b}))} required/>
                    <Field label="Sender Acc. No."  name="sender_account" value={form.sender_account} onChange={handleChange} placeholder="e.g. 3201 8475 2910" required/>
                    <Field label="Sender IFSC"      name="sender_ifsc"    value={form.sender_ifsc}    onChange={handleChange} placeholder="e.g. SBIN0001234"/>
                  </div>
                </div>
                <div>
                  <SectionHeader icon="🏦" title="Receiver Details"/>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 16px"}}>
                    <Field label="Receiver Name"    name="receiver_name"    value={form.receiver_name}    onChange={handleChange} placeholder="e.g. Amazon India" required/>
                    <Field label="Receiver Bank"    name="receiver_bank" type="select" value={form.receiver_bank} onChange={handleChange} options={BANKS.map(b=>({value:b,label:b}))}/>
                    <Field label="Receiver Acc. No."name="receiver_account" value={form.receiver_account} onChange={handleChange} placeholder="e.g. 9876 5432 1098" required/>
                    <Field label="Receiver IFSC"    name="receiver_ifsc"    value={form.receiver_ifsc}    onChange={handleChange} placeholder="e.g. HDFC0004521"/>
                  </div>
                </div>
                <div>
                  <SectionHeader icon="💳" title="Transaction Details"/>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"0 16px"}}>
                    <Field label="Transaction / Ref ID" name="transaction_id" value={form.transaction_id} onChange={handleChange} placeholder="e.g. UPI/TXN/2503071430" required hint="UPI Ref ID, NEFT/IMPS Ref No."/>
                    <Field label="Amount (₹)"       name="amount" type="number" value={form.amount} onChange={handleChange} placeholder="Enter amount" required/>
                    <Field label="Payment Mode"     name="payment_mode" type="select" value={form.payment_mode} onChange={handleChange} options={PAYMENT_MODES.map(m=>({value:m,label:m}))}/>
                    <Field label="Transaction Type" name="transaction_type" type="select" value={form.transaction_type} onChange={handleChange}
                      options={[{value:"purchase",label:"Purchase / Payment"},{value:"withdrawal",label:"Cash Withdrawal"},{value:"transfer",label:"Money Transfer"},{value:"bill",label:"Bill Payment"},{value:"recharge",label:"Mobile Recharge"},{value:"emi",label:"EMI Payment"},{value:"refund",label:"Refund"}]}/>
                    <Field label="Merchant / Payee" name="merchant_name" value={form.merchant_name} onChange={handleChange} placeholder="e.g. Swiggy, Amazon"/>
                    <Field label="Category"         name="merchant_category" type="select" value={form.merchant_category} onChange={handleChange} options={MERCHANT_CATEGORIES}/>
                    <Field label="Device Used"      name="device_type" type="select" value={form.device_type} onChange={handleChange}
                      options={[{value:"mobile",label:"Mobile Phone"},{value:"laptop",label:"Laptop / Desktop"},{value:"atm",label:"ATM Machine"},{value:"pos",label:"POS Terminal"},{value:"unknown",label:"Unknown Device ⚠️"}]}/>
                    <Field label="Transaction Hour" name="hour" type="number" value={form.hour} onChange={handleChange} placeholder="e.g. 14"/>
                  </div>
                  <Field label="Remarks" name="transaction_note" type="textarea" value={form.transaction_note} onChange={handleChange} placeholder="e.g. Monthly rent, Groceries (optional)"/>
                </div>
                <div>
                  <SectionHeader icon="⚠️" title="Risk Indicators"/>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:1}}>
                    <Field label="" name="saved_beneficiary"    type="checkbox" value={form.saved_beneficiary}    onChange={handleChange} placeholder="✅ Receiver is a saved beneficiary"/>
                    <Field label="" name="card_present"         type="checkbox" value={form.card_present}         onChange={handleChange} placeholder="✅ Card physically present"/>
                    <Field label="" name="location_mismatch"    type="checkbox" value={form.location_mismatch}    onChange={handleChange} placeholder="⚠️ Unusual location"/>
                    <Field label="" name="new_merchant"         type="checkbox" value={form.new_merchant}         onChange={handleChange} placeholder="⚠️ New / unknown merchant"/>
                    <Field label="" name="high_frequency"       type="checkbox" value={form.high_frequency}       onChange={handleChange} placeholder="⚠️ High transaction frequency"/>
                    <Field label="" name="international"        type="checkbox" value={form.international}        onChange={handleChange} placeholder="⚠️ International transaction"/>
                    <Field label="" name="online_transaction"   type="checkbox" value={form.online_transaction}   onChange={handleChange} placeholder="⚠️ Online transaction"/>
                    <Field label="" name="first_time_large_amt" type="checkbox" value={form.first_time_large_amt} onChange={handleChange} placeholder="⚠️ First-time large amount"/>
                    <Field label="" name="suspicious_link"      type="checkbox" value={form.suspicious_link}      onChange={handleChange} placeholder="🚨 Clicked suspicious link/QR"/>
                    <Field label="" name="otp_shared"           type="checkbox" value={form.otp_shared}           onChange={handleChange} placeholder="🚨 OTP / PIN shared"/>
                  </div>
                </div>
                {formError&&<div style={{background:"#ff386015",border:`1px solid ${C.fraud}44`,borderRadius:8,padding:"10px 14px",color:C.fraud,fontSize:13,fontWeight:600}}>⚠️ {formError}</div>}
                <button onClick={handleSubmit} disabled={loading} style={{width:"100%",padding:"14px",background:loading?C.muted:`linear-gradient(90deg,${C.safe},#00c896)`,border:"none",borderRadius:10,color:"#080b14",fontWeight:900,fontSize:15,cursor:loading?"not-allowed":"pointer",boxShadow:loading?"none":`0 0 28px ${C.safe}55`,transition:"all 0.2s",fontFamily:"inherit"}}>
                  {loading?"⏳  Analyzing Transaction...":"🔍  ANALYZE TRANSACTION"}
                </button>
              </div>
            </div>

            {/* RESULT PANEL */}
            <div>
              <h2 style={{marginBottom:16,fontSize:18,fontWeight:800}}>Analysis Result</h2>
              {result?(
                <div style={{display:"flex",flexDirection:"column",gap:14}}>
                  <div style={{background:C.card,border:`2px solid ${result.is_fraud?C.fraud:C.safe}66`,borderRadius:14,padding:20}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:14}}>
                      <div style={{fontSize:18,fontWeight:900,color:result.is_fraud?C.fraud:C.safe}}>{result.is_fraud?"🚨 FRAUD DETECTED":"✅ TRANSACTION SAFE"}</div>
                      <RiskBadge level={result.risk_level}/>
                    </div>
                    <FraudGauge value={result.fraud_probability}/>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginTop:12}}>
                      <div style={{background:"#00f5a011",border:`1px solid ${C.safe}33`,borderRadius:10,padding:"12px",textAlign:"center"}}>
                        <div style={{color:C.safe,fontSize:20,fontWeight:900,fontFamily:"monospace"}}>{result.safe_probability}%</div>
                        <div style={{color:C.muted,fontSize:11,marginTop:2}}>SAFE</div>
                      </div>
                      <div style={{background:"#ff386011",border:`1px solid ${C.fraud}33`,borderRadius:10,padding:"12px",textAlign:"center"}}>
                        <div style={{color:C.fraud,fontSize:20,fontWeight:900,fontFamily:"monospace"}}>{result.fraud_probability}%</div>
                        <div style={{color:C.muted,fontSize:11,marginTop:2}}>FRAUD RISK</div>
                      </div>
                    </div>
                    <div style={{marginTop:10,padding:"8px 12px",background:C.bg,borderRadius:8,fontSize:11,color:C.muted,fontFamily:"monospace"}}>
                      ML: {result.ml_score}% + Rule: +{result.rule_boost}% = {result.fraud_probability}%
                    </div>

                    {/* TXN ID VALIDITY */}
                    {result.txn_id_valid !== undefined && (
                      <div style={{marginTop:10,padding:"8px 12px",background:result.txn_id_valid?"#00f5a011":"#ff386011",border:`1px solid ${result.txn_id_valid?C.safe:C.fraud}33`,borderRadius:8,display:"flex",alignItems:"center",gap:8,fontSize:12}}>
                        <span>{result.txn_id_valid?"✅":"⚠️"}</span>
                        <span style={{color:result.txn_id_valid?C.safe:C.fraud,fontWeight:700}}>
                          TXN ID: {result.txn_id_valid?"Valid format":"Suspicious format detected"}
                        </span>
                        {result.txn_id_risk>0&&<span style={{color:C.muted,marginLeft:"auto",fontFamily:"monospace"}}>+{result.txn_id_risk}% risk</span>}
                      </div>
                    )}

                    {/* FEATURE 1: PDF DOWNLOAD BUTTON */}
                    <button onClick={handleDownloadPDF} disabled={pdfLoading} style={{width:"100%",marginTop:12,padding:"11px",background:pdfLoading?"#1f2937":`linear-gradient(90deg,${C.accent}44,${C.accent}22)`,border:`1px solid ${C.accent}`,borderRadius:10,color:C.accent,fontWeight:800,fontSize:13,cursor:pdfLoading?"not-allowed":"pointer",fontFamily:"inherit",transition:"all 0.2s"}}>
                      {pdfLoading?"⏳ Generating PDF...":"📄 Download PDF Report"}
                    </button>
                  </div>

                  <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:18}}>
                    <div style={{color:C.muted,fontSize:11,letterSpacing:2,marginBottom:12,textTransform:"uppercase"}}>📋 Transaction Summary</div>
                    {[
                      ["🔖 Ref ID",  result.transaction_id],
                      ["👤 Sender",  result.cardholder_name],
                      ["🏦 Bank",    result.bank_name],
                      ["💳 From",    result.sender_account?"XXXX "+result.sender_account.slice(-4):"—"],
                      ["📤 To",      result.receiver_name],
                      ["💳 To A/C",  result.receiver_account?"XXXX "+result.receiver_account.slice(-4):"—"],
                      ["💰 Amount",  `₹${Number(result.amount||0).toLocaleString()}`],
                      ["📱 Mode",    result.payment_mode],
                    ].map(([k,v])=>v&&v!=="—"?(
                      <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"6px 0",borderBottom:`1px solid ${C.border}`,fontSize:12}}>
                        <span style={{color:C.muted}}>{k}</span>
                        <span style={{color:C.text,fontWeight:600,fontFamily:k.includes("A/C")||k.includes("ID")?"monospace":"inherit"}}>{v}</span>
                      </div>
                    ):null)}
                  </div>

                  {result.risk_factors?.length>0&&(
                    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:18}}>
                      <div style={{color:C.muted,fontSize:11,letterSpacing:2,marginBottom:10,textTransform:"uppercase"}}>⚠️ Risk Factors</div>
                      {result.risk_factors.map((f,i)=>(
                        <div key={i} style={{display:"flex",alignItems:"center",gap:10,marginBottom:7,fontSize:12,color:C.high,background:"#ff6b3511",borderRadius:8,padding:"7px 12px"}}><span>▶</span>{f}</div>
                      ))}
                    </div>
                  )}

                  {result.is_fraud&&(
                    <div style={{background:"#ff073a0a",border:`1px solid ${C.critical}33`,borderRadius:14,padding:18}}>
                      <div style={{color:C.critical,fontSize:13,fontWeight:800,marginBottom:10}}>🛑 Immediate Action Required</div>
                      {["Call your bank helpline immediately","Do NOT share OTP, PIN, or CVV","Report to cybercrime.gov.in or call 1930","Change your banking password & MPIN","Check all recent transactions"].map((tip,i)=>(
                        <div key={i} style={{fontSize:12,color:C.muted,marginBottom:6,display:"flex",gap:8}}><span style={{color:C.critical}}>•</span>{tip}</div>
                      ))}
                    </div>
                  )}
                </div>
              ):(
                <div style={{background:C.card,border:`1px dashed ${C.border}`,borderRadius:14,padding:70,textAlign:"center"}}>
                  <div style={{fontSize:48,marginBottom:14}}>🔐</div>
                  <div style={{color:C.muted,fontSize:14,lineHeight:1.8}}>Fill in the transaction details<br/>and click <b style={{color:C.safe}}>Analyze Transaction</b></div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══ DASHBOARD ══ */}
        {tab==="dashboard"&&(
          <div>
            <h2 style={{marginBottom:20,fontSize:18,fontWeight:800}}>Analytics Dashboard</h2>
            {stats?(<>
              <div style={{display:"flex",gap:14,flexWrap:"wrap",marginBottom:22}}>
                <StatCard label="Total" value={stats.total}       color={C.safe}/>
                <StatCard label="Fraudulent" value={stats.frauds} sub={`${stats.fraud_rate}% rate`} color={C.fraud}/>
                <StatCard label="Legitimate" value={stats.legitimate} color={C.accent}/>
                <StatCard label="Fraud Amount" value={`₹${Number(stats.fraud_amount).toLocaleString()}`} color={C.warn}/>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
                <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:22}}>
                  <div style={{fontWeight:700,marginBottom:14}}>Fraud vs Legitimate</div>
                  <ResponsiveContainer width="100%" height={190}>
                    <PieChart><Pie data={[{name:"Legitimate",value:stats.legitimate},{name:"Fraudulent",value:stats.frauds}]} cx="50%" cy="50%" outerRadius={72} dataKey="value" paddingAngle={3}><Cell fill={C.safe}/><Cell fill={C.fraud}/></Pie><Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8}}/><Legend formatter={v=><span style={{color:C.text,fontSize:12}}>{v}</span>}/></PieChart>
                  </ResponsiveContainer>
                </div>
                <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:22}}>
                  <div style={{fontWeight:700,marginBottom:14}}>Risk Distribution</div>
                  <ResponsiveContainer width="100%" height={190}>
                    <BarChart data={Object.entries(stats.risk_distribution).map(([k,v])=>({name:k,count:v}))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
                      <XAxis dataKey="name" tick={{fill:C.muted,fontSize:11}}/><YAxis tick={{fill:C.muted,fontSize:10}}/>
                      <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8}}/>
                      <Bar dataKey="count" radius={[4,4,0,0]}>{Object.keys(stats.risk_distribution).map((k,i)=><Cell key={i} fill={{LOW:C.low,MEDIUM:C.medium,HIGH:C.high,CRITICAL:C.critical}[k]}/>)}</Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:22,gridColumn:"span 2"}}>
                  <div style={{fontWeight:700,marginBottom:14}}>Fraud Probability Trend</div>
                  <ResponsiveContainer width="100%" height={180}>
                    <AreaChart data={history.slice(-20).map((t,i)=>({name:`#${i+1}`,fraud:t.fraud_probability,safe:t.safe_probability}))}>
                      <defs>
                        <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.fraud} stopOpacity={0.3}/><stop offset="95%" stopColor={C.fraud} stopOpacity={0}/></linearGradient>
                        <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.safe} stopOpacity={0.3}/><stop offset="95%" stopColor={C.safe} stopOpacity={0}/></linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
                      <XAxis dataKey="name" tick={{fill:C.muted,fontSize:10}}/><YAxis tick={{fill:C.muted,fontSize:10}}/>
                      <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8}}/>
                      <Area type="monotone" dataKey="safe"  stroke={C.safe}  fill="url(#g2)" strokeWidth={2}/>
                      <Area type="monotone" dataKey="fraud" stroke={C.fraud} fill="url(#g1)" strokeWidth={2}/>
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>):<div style={{textAlign:"center",color:C.muted,padding:80}}>No data yet. Analyze some transactions first.</div>}
          </div>
        )}

        {/* ══ ADVANCED ANALYTICS ══ */}
        {tab==="analytics"&&<AdvancedAnalytics stats={stats} history={history}/>}

        {/* ══ HISTORY ══ */}
        {tab==="history"&&(
          <div>
            <h2 style={{marginBottom:20,fontSize:18,fontWeight:800}}>Transaction History</h2>
            {history.length===0?<div style={{textAlign:"center",color:C.muted,padding:80}}>No transactions yet.</div>:(
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"separate",borderSpacing:"0 8px",minWidth:950}}>
                  <thead><tr style={{color:C.muted,fontSize:11,letterSpacing:2,textTransform:"uppercase"}}>{["Ref ID","Sender","Bank","To","Amount","Mode","ML","Boost","Total","Risk","Status"].map(h=><th key={h} style={{textAlign:"left",padding:"7px 10px",fontWeight:600}}>{h}</th>)}</tr></thead>
                  <tbody>
                    {[...history].reverse().map((t,i)=>(
                      <tr key={i} style={{background:C.card}}>
                        <td style={{padding:"10px 10px",fontSize:11,fontFamily:"monospace",color:C.muted,borderRadius:"8px 0 0 8px"}}>{t.transaction_id||"—"}</td>
                        <td style={{padding:"10px 10px",fontSize:12}}>{t.cardholder_name||"—"}</td>
                        <td style={{padding:"10px 10px",fontSize:11,color:C.muted}}>{t.bank_name||"—"}</td>
                        <td style={{padding:"10px 10px",fontSize:12}}>{t.receiver_name||"—"}</td>
                        <td style={{padding:"10px 10px",fontWeight:700}}>₹{Number(t.amount||0).toLocaleString()}</td>
                        <td style={{padding:"10px 10px",fontSize:11,color:C.muted}}>{t.payment_mode||"—"}</td>
                        <td style={{padding:"10px 10px",fontSize:11,fontFamily:"monospace",color:C.accent}}>{t.ml_score||"?"}%</td>
                        <td style={{padding:"10px 10px",fontSize:11,fontFamily:"monospace",color:C.warn}}>+{t.rule_boost||0}%</td>
                        <td style={{padding:"10px 10px",fontFamily:"monospace",fontWeight:700,color:t.fraud_probability>50?C.fraud:C.safe}}>{t.fraud_probability}%</td>
                        <td style={{padding:"10px 10px"}}><RiskBadge level={t.risk_level}/></td>
                        <td style={{padding:"10px 10px",borderRadius:"0 8px 8px 0"}}><span style={{color:t.is_fraud?C.fraud:C.safe,fontWeight:700,fontSize:12}}>{t.is_fraud?"🚨 FRAUD":"✅ SAFE"}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ══ MODEL ══ */}
        {tab==="model"&&(
          <div>
            <h2 style={{marginBottom:20,fontSize:18,fontWeight:800}}>Model Performance</h2>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
              <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:24}}>
                <div style={{fontWeight:800,fontSize:15,marginBottom:16}}>📈 Accuracy Metrics</div>
                {stats?.model_metrics&&(<>
                  <MetricBar label="Accuracy"           value={stats.model_metrics.accuracy}        color={C.safe}/>
                  <MetricBar label="AUC-ROC"            value={stats.model_metrics.auc_roc}         color={C.accent}/>
                  <MetricBar label="Avg Precision (AP)" value={stats.model_metrics.avg_precision}   color={C.warn}/>
                  <MetricBar label="Precision (Fraud)"  value={stats.model_metrics.precision_fraud} color={C.high}/>
                  <MetricBar label="Recall (Fraud)"     value={stats.model_metrics.recall_fraud}    color={C.medium}/>
                  <MetricBar label="F1-Score"           value={stats.model_metrics.f1_fraud}        color={C.safe}/>
                  <div style={{marginTop:14,padding:"10px 14px",background:"#00f5a008",border:`1px solid ${C.safe}22`,borderRadius:8}}>
                    <div style={{color:C.muted,fontSize:11,marginBottom:3}}>OPTIMAL THRESHOLD</div>
                    <div style={{color:C.safe,fontFamily:"monospace",fontSize:18,fontWeight:700}}>{stats.model_metrics.threshold}</div>
                  </div>
                </>)}
              </div>
              <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:24}}>
                <div style={{fontWeight:800,fontSize:15,marginBottom:16}}>🔢 Confusion Matrix</div>
                {stats?.model_metrics?.confusion_matrix&&(()=>{
                  const cm=stats.model_metrics.confusion_matrix;
                  const tn=cm[0][0],fp=cm[0][1],fn=cm[1][0],tp=cm[1][1];
                  return(<>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginBottom:16}}>
                      {[{label:"TN",val:tn,desc:"Correctly Legit",color:C.safe},{label:"FP",val:fp,desc:"Legit → Fraud",color:C.warn},{label:"FN",val:fn,desc:"Fraud missed",color:C.high},{label:"TP",val:tp,desc:"Fraud caught",color:C.safe}].map(({label,val,desc,color})=>(
                        <div key={label} style={{background:color+"11",border:`1px solid ${color}33`,borderRadius:10,padding:14}}>
                          <div style={{color,fontSize:20,fontWeight:900,fontFamily:"monospace"}}>{val?.toLocaleString()}</div>
                          <div style={{color:C.text,fontSize:11,fontWeight:700,marginTop:3}}>{label}</div>
                          <div style={{color:C.muted,fontSize:10,marginTop:2}}>{desc}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{color:C.muted,fontSize:11,lineHeight:1.8,padding:"10px",background:C.bg,borderRadius:8}}>
                      <div><b style={{color:C.text}}>Dataset:</b> {stats.model_metrics.dataset}</div>
                      <div><b style={{color:C.text}}>Train:</b> {stats.model_metrics.train_samples?.toLocaleString()}</div>
                      <div><b style={{color:C.text}}>Test:</b> {stats.model_metrics.test_samples?.toLocaleString()}</div>
                    </div>
                  </>);
                })()}
              </div>
              <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:24,gridColumn:"span 2"}}>
                <div style={{fontWeight:800,fontSize:15,marginBottom:16}}>⚙️ Pipeline & Architecture</div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14}}>
                  {[
                    {n:"01",t:"Kaggle Data",       d:"284,807 real credit card transactions, V1-V28 PCA",  i:"📂"},
                    {n:"02",t:"Feature Eng.",       d:"RobustScaler + 10 polynomial & interaction features",i:"🔧"},
                    {n:"03",t:"SMOTE Balancing",    d:"Minority fraud class oversampled to match legit",    i:"⚖️"},
                    {n:"04",t:"Ensemble Training",  d:"RF(300)+ET(300)+GB(200)+LR soft-vote weighted",      i:"🧠"},
                    {n:"05",t:"Rule-Based Boost",   d:"OTP/link/device signals add on top of ML score",    i:"📏"},
                    {n:"06",t:"Auto-Retrain",       d:"Background retrain every 10 new transactions",       i:"🔄"},
                  ].map(({n,t,d,i})=>(
                    <div key={n} style={{padding:"14px",background:C.bg,borderRadius:10,borderLeft:`3px solid ${C.safe}`}}>
                      <div style={{color:C.safe,fontSize:10,fontFamily:"monospace",marginBottom:5}}>STEP {n} {i}</div>
                      <div style={{fontWeight:700,fontSize:12,marginBottom:5}}>{t}</div>
                      <div style={{color:C.muted,fontSize:11,lineHeight:1.5}}>{d}</div>
                    </div>
                  ))}
                </div>
                <button onClick={handleRetrain} disabled={loading} style={{marginTop:18,padding:"11px 24px",background:"none",border:`1px solid ${C.accent}`,color:C.accent,borderRadius:8,cursor:"pointer",fontWeight:700,fontSize:13,fontFamily:"inherit"}}>
                  {loading?"⏳ Retraining...":"🔄 Retrain Model"}
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
