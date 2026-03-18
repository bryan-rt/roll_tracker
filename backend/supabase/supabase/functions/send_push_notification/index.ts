// send_push_notification — Supabase Edge Function
// Triggered by DB webhooks on the clips table.
// Sends FCM V1 push notifications for clip_ready and collision_claim events.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { create } from "https://deno.land/x/djwt@v3.0.2/mod.ts";

interface WebhookPayload {
  type: "INSERT" | "UPDATE";
  table: string;
  record: Record<string, unknown>;
  old_record: Record<string, unknown> | null;
}

// --- FCM V1 Auth ---

async function getAccessToken(serviceAccount: {
  client_email: string;
  private_key: string;
  token_uri: string;
}): Promise<string> {
  const now = Math.floor(Date.now() / 1000);

  // Import the private key for signing
  const pemContent = serviceAccount.private_key
    .replace(/-----BEGIN PRIVATE KEY-----/, "")
    .replace(/-----END PRIVATE KEY-----/, "")
    .replace(/\n/g, "");
  const binaryKey = Uint8Array.from(atob(pemContent), (c) => c.charCodeAt(0));

  const cryptoKey = await crypto.subtle.importKey(
    "pkcs8",
    binaryKey,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"]
  );

  const jwt = await create(
    { alg: "RS256", typ: "JWT" },
    {
      iss: serviceAccount.client_email,
      scope: "https://www.googleapis.com/auth/firebase.messaging",
      aud: serviceAccount.token_uri,
      exp: now + 3600,
      iat: now,
    },
    cryptoKey
  );

  const tokenRes = await fetch(serviceAccount.token_uri, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`,
  });

  const tokenData = await tokenRes.json();
  return tokenData.access_token;
}

async function sendFcmMessage(
  accessToken: string,
  projectId: string,
  token: string,
  title: string,
  body: string,
  data?: Record<string, string>
): Promise<void> {
  const url = `https://fcm.googleapis.com/v1/projects/${projectId}/messages:send`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: {
        token,
        notification: { title, body },
        data: data || {},
      },
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    console.error(`FCM send failed for token ${token.slice(0, 10)}...: ${err}`);
  }
}

// --- Main handler ---

serve(async (req) => {
  try {
    const payload: WebhookPayload = await req.json();
    const record = payload.record;

    // Initialize Supabase client (service role for reading device_tokens)
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Parse FCM service account
    const fcmJson = Deno.env.get("FCM_SERVICE_ACCOUNT");
    if (!fcmJson) {
      console.error("FCM_SERVICE_ACCOUNT not set");
      return new Response("ok", { status: 200 });
    }
    const serviceAccount = JSON.parse(fcmJson);
    const projectId = serviceAccount.project_id;

    // Get FCM access token
    const accessToken = await getAccessToken(serviceAccount);

    // Determine notification type
    const status = record.status as string;
    const fighterAProfileId = record.fighter_a_profile_id as string | null;
    const fighterBProfileId = record.fighter_b_profile_id as string | null;
    const clipId = record.id as string;

    let profileIds: string[] = [];
    let title = "";
    let body = "";
    let data: Record<string, string> = { clip_id: clipId };

    if (status === "collision_flagged") {
      // Collision claim: notify athletes with matching tags at the gym
      title = "Clip may be yours";
      body = "A clip may belong to you — check Unlinked Clips.";
      data.type = "collision_claim";

      const tagA = record.fighter_a_tag_id as number | null;
      const tagB = record.fighter_b_tag_id as number | null;
      const videoId = record.video_id as string | null;

      if (videoId) {
        // Get gym_id from videos table
        const { data: videoRow } = await supabase
          .from("videos")
          .select("gym_id")
          .eq("id", videoId)
          .maybeSingle();

        const gymId = videoRow?.gym_id;
        if (gymId) {
          const tags = [tagA, tagB].filter((t) => t != null) as number[];
          if (tags.length > 0) {
            const { data: checkins } = await supabase
              .from("gym_checkins")
              .select("profile_id, profiles!inner(tag_id)")
              .eq("gym_id", gymId)
              .eq("is_active", true)
              .in("profiles.tag_id", tags);

            if (checkins) {
              profileIds = checkins.map((c: Record<string, unknown>) => c.profile_id as string);
            }
          }
        }
      }
    } else if (fighterAProfileId || fighterBProfileId) {
      // Clips ready: notify resolved profiles
      title = "Match clip ready";
      body = "Your match clip is ready to watch.";
      data.type = "clips_ready";

      if (fighterAProfileId) profileIds.push(fighterAProfileId);
      if (fighterBProfileId) profileIds.push(fighterBProfileId);
    }

    // Deduplicate
    profileIds = [...new Set(profileIds)];

    if (profileIds.length === 0) {
      return new Response("ok", { status: 200 });
    }

    // Fetch device tokens
    const { data: tokens } = await supabase
      .from("device_tokens")
      .select("token")
      .in("profile_id", profileIds);

    if (!tokens || tokens.length === 0) {
      console.log(`No device tokens for profiles: ${profileIds.join(", ")}`);
      return new Response("ok", { status: 200 });
    }

    // Send notifications
    for (const { token } of tokens) {
      try {
        await sendFcmMessage(accessToken, projectId, token, title, body, data);
      } catch (e) {
        console.error(`Failed to send to token ${token.slice(0, 10)}...: ${e}`);
      }
    }

    return new Response("ok", { status: 200 });
  } catch (e) {
    console.error(`send_push_notification error: ${e}`);
    return new Response("ok", { status: 200 }); // never fail the webhook
  }
});
