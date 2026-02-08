document.addEventListener("DOMContentLoaded", () => {
    gsap.registerPlugin(ScrollTrigger);

    // 1. SMOOTH SCROLLING
    const lenis = new Lenis();
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((t) => lenis.raf(t * 1000));

    // 2. GLOBE ENGINE (With Trends and Fixed Projection)
    const gCanvas = document.getElementById("globeCanvas");
    const gCtx = gCanvas.getContext("2d");
    let gTime = 0;

    function resizeG() {
        gCanvas.width = window.innerWidth;
        gCanvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resizeG);
    resizeG();

    const particles = [];
    const COUNT = 1400;
    const RADIUS = Math.min(window.innerWidth, window.innerHeight) * 0.42;
    const DEPTH = 1000;

    for (let i = 0; i < COUNT; i++) {
        const u = Math.random();
        const v = Math.random();
        const theta = 2 * Math.PI * u;
        const phi = Math.acos(2 * v - 1);
        particles.push({
            angle: Math.random() * Math.PI * 2,
            orbitR: 520 + Math.random() * 260,
            speed: Math.random() * 0.005 + 0.002,
            tx: RADIUS * Math.sin(phi) * Math.cos(theta),
            ty: RADIUS * Math.cos(phi),
            tz: RADIUS * Math.sin(phi) * Math.sin(theta),
            color: Math.random() > 0.5 ? "#22c55e" : "#ef4444"
        });
    }

    function project(x, y, z) {
        const s = DEPTH / (DEPTH + z);
        return { x: (window.innerWidth * 0.72) + x * s, y: (window.innerHeight * 0.5) + y * s, s: s };
    }

    function drawTrendLine(offset, color, alpha) {
        gCtx.beginPath();
        for (let i = 0; i <= 120; i++) {
            const a = (i / 120) * Math.PI * 2;
            const wave = Math.sin(a * 3 + gTime + offset) * 18;
            let x = (RADIUS + wave) * Math.cos(a);
            let y = Math.sin(a * 2 + offset) * 24;
            let z = (RADIUS + wave) * Math.sin(a);

            const ry = gTime * 0.3;
            const rx = Math.PI / 6;
            let y1 = y * Math.cos(rx) - z * Math.sin(rx);
            let z1 = y * Math.sin(rx) + z * Math.cos(rx);
            let x2 = x * Math.cos(ry) - z1 * Math.sin(ry);
            let z2 = x * Math.sin(ry) + z1 * Math.cos(ry);

            const p = project(x2, y1, z2);
            if (i === 0) gCtx.moveTo(p.x, p.y);
            else gCtx.lineTo(p.x, p.y);
        }
        gCtx.strokeStyle = color;
        gCtx.globalAlpha = alpha;
        gCtx.lineWidth = 1.2;
        gCtx.stroke();
    }

    function animateGlobe() {
        gCtx.clearRect(0, 0, gCanvas.width, gCanvas.height);
        gTime += 0.015;
        const morph = Math.min(1, Math.max(0, (gTime - 1) / 3));
        const linesAlpha = Math.max(0, Math.min(0.8, (gTime - 4) / 2));

        particles.forEach(p => {
            p.angle += p.speed;
            const ox = Math.cos(p.angle) * p.orbitR;
            const oy = Math.sin(p.angle * 0.9) * p.orbitR * 0.5;
            const oz = Math.sin(p.angle) * 300;

            const x = ox + (p.tx - ox) * morph;
            const y = oy + (p.ty - oy) * morph;
            const z = oz + (p.tz - oz) * morph;

            const ry = gTime * 0.3;
            const rx = Math.PI / 6;
            let y1 = y * Math.cos(rx) - z * Math.sin(rx);
            let z1 = y * Math.sin(rx) + z * Math.cos(rx);
            let x2 = x * Math.cos(ry) - z1 * Math.sin(ry);
            let z2 = x * Math.sin(ry) + z1 * Math.cos(ry);

            const pos = project(x2, y1, z2);
            gCtx.globalAlpha = pos.s;
            gCtx.fillStyle = p.color;
            gCtx.beginPath();
            gCtx.arc(pos.x, pos.y, 2.2 * pos.s, 0, Math.PI * 2);
            gCtx.fill();
        });

        if (linesAlpha > 0) {
            drawTrendLine(0, "#22c55e", linesAlpha);
            drawTrendLine(2.4, "#ef4444", linesAlpha);
        }
        requestAnimationFrame(animateGlobe);
    }
    animateGlobe();

    // 3. TRIANGLE GALLERY ENGINE
    const outC = document.querySelector(".outline-layer");
    const fillC = document.querySelector(".fill-layer");
    const oCtx = outC.getContext("2d");
    const fCtx = fillC.getContext("2d");
    const triStates = [];

    function initTris() {
        const dpr = window.devicePixelRatio || 1;
        [outC, fillC].forEach(c => {
            c.width = window.innerWidth * dpr;
            c.height = window.innerHeight * dpr;
            c.style.width = window.innerWidth + "px";
            c.style.height = window.innerHeight + "px";
        });
        oCtx.scale(dpr, dpr); fCtx.scale(dpr, dpr);
        for(let r=0; r<15; r++) {
            for(let c=0; c<30; c++) {
                triStates.push({ r, c, order: Math.random(), scale: 0 });
            }
        }
    }
    initTris();

    function drawTri(ctx, x, y, s, flip, fill) {
        const size = 150; const h = size / 2;
        ctx.save();
        ctx.translate(x, y); ctx.scale(s, s); ctx.translate(-x, -y);
        ctx.beginPath();
        if(!flip) { ctx.moveTo(x, y-h); ctx.lineTo(x+h, y+h); ctx.lineTo(x-h, y+h); }
        else { ctx.moveTo(x, y+h); ctx.lineTo(x+h, y-h); ctx.lineTo(x-h, y-h); }
        ctx.closePath();
        if(fill) { ctx.fillStyle = "#ff6b00"; ctx.fill(); }
        else { ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.stroke(); }
        ctx.restore();
    }

    ScrollTrigger.create({
        trigger: ".sticky-gallery",
        start: "top top",
        end: "+=3000",
        pin: true,
        onUpdate: (self) => {
            oCtx.clearRect(0,0,window.innerWidth, window.innerHeight);
            fCtx.clearRect(0,0,window.innerWidth, window.innerHeight);
            const p = self.progress;
            const animP = p < 0.65 ? 0 : (p - 0.65) / 0.35;
            
            triStates.forEach(s => {
                const x = s.c * 75 + 75 - (p * 200);
                const y = s.r * 150 + 75;
                const flip = (s.r + s.c) % 2 !== 0;
                drawTri(oCtx, x, y, 1, flip, false);
                if(s.order < animP) {
                    s.scale = Math.min(1, s.scale + 0.15);
                    drawTri(fCtx, x, y, s.scale, flip, true);
                }
            });

            const cards = document.querySelector(".cards-container");
            gsap.set(cards, { x: -Math.min(p/0.654, 1) * window.innerWidth * 2 });
        }
    });

    // 4. TEXT INTERACTION ENGINE
    window.addEventListener("scroll", () => {
        const s = document.querySelector(".scroll-section");
        const r = s.getBoundingClientRect();
        if(r.top < window.innerHeight && r.bottom > 0) {
            const p = 1 - (r.bottom / (window.innerHeight + r.height));
            document.querySelectorAll(".img-box").forEach(b => {
                b.style.width = (70 + p * 280) + "px";
            });
        }
    });
});