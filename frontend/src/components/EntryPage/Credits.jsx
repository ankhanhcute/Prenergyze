const Credits = () => {
  const teamMembers = [
    {
      role: "Team Leads",
      members: ["Adrian Morton", "Leonardo Herrera"]
    },
    {
      role: "Machine Learning",
      members: ["Khang Ho", "Khanh Truong", "Ethan Rodriguez", "Rhode Sanchez"]
    },
    {
      role: "Data Analysis",
      members: ["Annette Garcia", "Gabriella Hernandez", "Julian Novak"]
    },
    {
      role: "Backend Development",
      members: ["Adrian Morton", "Leonardo Herrera"]
    },
    {
      role: "Frontend Development",
      members: ["Adrian Morton"]
    }
  ];

  return (
    <div className="card" style={{ textAlign: 'center' }}>
      <h2 style={{ 
        fontSize: '24px', 
        marginBottom: '30px', 
        color: '#f3f4f6',
        borderBottom: '1px solid rgba(245, 158, 11, 0.2)',
        paddingBottom: '15px',
        display: 'inline-block',
        paddingLeft: '40px',
        paddingRight: '40px'
      }}>
        Project Team
      </h2>
      
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', 
        gap: '20px',
        padding: '0 10px'
      }}>
        {teamMembers.map((group, index) => (
          <div key={index} style={{ marginBottom: '10px' }}>
            <h3 style={{ 
              color: '#f59e0b', 
              fontSize: '16px',  
              fontWeight: '600',
              marginBottom: '12px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              {group.role}
            </h3>
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              gap: '6px',
              color: '#d1d5db'
            }}>
              {group.members.map((member, mIndex) => (
                <span key={mIndex} style={{ fontSize: '15px' }}>{member}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Credits;
